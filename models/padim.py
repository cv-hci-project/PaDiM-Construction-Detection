import torch
from torch.nn import Parameter

from models import PaDiMBase
from utils.utils import get_embedding


class PaDiM(PaDiMBase):

    def __init__(self, params: dict, backbone_params: dict, device):
        super().__init__(params=params, backbone_params=backbone_params, device=device)

        self.means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings), device=self.device),
            requires_grad=False
        )

        self.covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings),
                        device=self.device),
            requires_grad=False
        )

        self.learned_means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings), device=self.device),
            requires_grad=False
        )

        self.learned_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings),
                        device=self.device),
            requires_grad=False
        )

        self.learned_inverse_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings),
                        device=self.device),
            requires_grad=False
        )

        # feature1, _, _ = self.backbone(test)
        # self.number_of_patches = feature1.size()[2]

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

    def forward(self, batch, min_max_norm: bool = True):
        x, _ = batch
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
