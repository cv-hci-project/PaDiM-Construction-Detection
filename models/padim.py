from scipy.ndimage import gaussian_filter

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F

from backbones import backbone_models
from utils.utils import get_embedding


class PaDiM(Module):

    def __init__(self, params: dict, device):
        super().__init__()

        self.device = device

        self.params = params

        self.backbone = backbone_models[params["backbone"]]()
        self.backbone.to(device)

        self.number_of_patches = self.backbone.number_of_patches
        self.number_of_embeddings = params["number_of_embeddings"]

        self.means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.embedding_ids = Parameter(
            torch.randperm(self.backbone.embeddings_size)[:self.number_of_embeddings].to(self.device),
            requires_grad=False
        )

        self.n = 0

        self.learned_means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.learned_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.learned_inverse_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.crop_size = params["crop_size"]

    def calculate_means_and_covariances(self):
        self.learned_means = Parameter(self.means.clone().to(self.device), requires_grad=False)
        self.learned_covariances = Parameter(self.covariances.clone().to(self.device), requires_grad=False)

        epsilon = 0.01

        identity = torch.eye(self.number_of_embeddings).to(self.device)
        self.learned_means /= self.n

        for i in range(self.number_of_patches):
            self.learned_covariances[i, :, :] -= self.n * torch.outer(self.learned_means[i, :],
                                                                      self.learned_means[i, :])
            self.learned_covariances[i, :, :] /= self.n - 1  # corrected covariance
            self.learned_covariances[i, :, :] += epsilon * identity  # constant term

        self.learned_inverse_covariances = Parameter(torch.linalg.inv(self.learned_covariances), requires_grad=False)

        # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
        #  variables to track the batches)

    def forward(self, x, min_max_norm: bool = True):
        x = x.to(self.device)

        with torch.no_grad():
            features_1, features_2, features_3 = self.backbone(x)

        embeddings = get_embedding(features_1, features_2, features_3, self.embedding_ids, self.device)
        b, c, h, w = embeddings.size()

        embeddings = embeddings.view(-1, self.number_of_embeddings, self.number_of_patches)

        if self.training:

            for i in range(self.number_of_patches):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(b):
                    self.covariances[i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=0)  # c
            self.n += b  # number of images
        else:
            return self.calculate_score_map(embeddings, (b, c, h, w), min_max_norm=min_max_norm)

    # def _test_batched_version(self, embeddings, embedding_dimensions: tuple):
    #     b, c, h, w = embedding_dimensions
    #
    #     for i in range(self.number_of_patches):
    #         patch_embeddings = embeddings[:, :, i]  # b * c
    #
    #         self.second_covariances[i, :, :] = torch.einsum('bi,bj->bij', patch_embeddings, patch_embeddings).sum(dim=0)
    #         self.second_means[i, :] += patch_embeddings.sum(dim=0)  # c
    #
    #     self.n2 += b  # number of images

    def _calculate_dist_list(self, embedding, embedding_dimensions: tuple):
        b, c, h, w = embedding_dimensions

        delta = embedding.transpose(2, 1) - self.learned_means.unsqueeze(0)

        # Calculates the mahalanobis distance in a batched manner
        batched_tensor_result = torch.sqrt(
            torch.einsum('bij,ijk,bik->bi', delta, self.learned_inverse_covariances, delta)
        )

        return batched_tensor_result.view((b, h, w))

    def calculate_score_map(self, embedding, embedding_dimensions: tuple, min_max_norm: bool) -> torch.Tensor:
        dist_list = self._calculate_dist_list(embedding, embedding_dimensions)

        # Upsample
        score_map = F.interpolate(dist_list.unsqueeze(1), size=self.crop_size, mode='bilinear',
                                  align_corners=False).squeeze().numpy()

        # Apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        if min_max_norm:
            max_score = score_map.max()
            min_score = score_map.min()
            scores = (score_map - min_score) / (max_score - min_score)
        else:
            scores = score_map

        return torch.Tensor(scores).to(self.device)
