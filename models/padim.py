from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
import torch
import numpy as np
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
        self.num_of_components = 3

        self.means_d = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.covariances_d = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.means_p = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.covariances_p = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings,
                         self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.means_w = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.covariances_w = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings,
                         self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.embedding_ids = Parameter(
            torch.randperm(self.backbone.embeddings_size)[:self.number_of_embeddings].to(self.device),
            requires_grad=False
        )

        self.n_d = 0
        self.n_p = 0
        self.n_w = 0

        self.learned_means = Parameter(
            torch.zeros((self.number_of_patches, self.num_of_components, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.learned_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.num_of_components, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.learned_inverse_covariances = Parameter(
            torch.zeros((self.number_of_patches, self.num_of_components, self.number_of_embeddings, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )
        self.crop_size = params["crop_size"]

    def calculate_means_and_covariances(self):
        # for i in range(self.number_of_patches):
        #     pixel_data = self.data[i, :, :]
        #     self.gm = GaussianMixture(n_components=3, random_state=0).fit(pixel_data)
        #     self.learned_means[i, :, :] = self.gm.means_
        #     self.learned_covariances[i, :, :, :] = self.gm.covariances_
        #     self.learned_inverse_covariances = Parameter(torch.linalg.inv(self.learned_covariances),
        #                                                  requires_grad=False)

        epsilon = 0.01
        identity = torch.eye(self.number_of_embeddings).to(self.device)
        self.learned_means[:, 0, :] = self.means_d / self.n_d
        self.learned_means[:, 1, :] = self.means_p / self.n_p
        self.learned_means[:, 2, :] = self.means_w / self.n_w
        for j in range(self.number_of_patches):
            self.learned_covariances[j, 0, :, :] = self.covariances_d[j, :, :] - self.n_d * torch.outer(self.learned_means[j, 0, :],
                                                                           self.learned_means[j, 0, :])
            self.learned_covariances[j, 0, :, :] /= self.n_d - 1  # corrected covariance
            self.learned_covariances[j, 0, :, :] += epsilon * identity  # constant term

            self.learned_covariances[j, 1, :, :] = self.covariances_p[j, :, :] - self.n_p * torch.outer(self.learned_means[j, 1, :],
                                                                           self.learned_means[j, 1, :])
            self.learned_covariances[j, 1, :, :] /= self.n_p - 1  # corrected covariance
            self.learned_covariances[j, 1, :, :] += epsilon * identity  # constant term

            self.learned_covariances[j, 2, :, :] = self.covariances_w[j, :, :] - self.n_w * torch.outer(self.learned_means[j, 2, :],
                                                                           self.learned_means[j, 2, :])
            self.learned_covariances[j, 2, :, :] /= self.n_w - 1  # corrected covariance
            self.learned_covariances[j, 2, :, :] += epsilon * identity  # constant term

        self.learned_inverse_covariances = Parameter(torch.linalg.inv(self.learned_covariances), requires_grad=False)

        # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
        #  variables to track the batches)

    def forward(self, x, category_data, min_max_norm: bool = True):
        x = x.to(self.device)

        with torch.no_grad():
            features_1, features_2, features_3 = self.backbone(x)

        embeddings = get_embedding(features_1, features_2, features_3, self.embedding_ids, self.device)
        b, c, h, w = embeddings.size()

        embeddings = embeddings.view(-1, self.number_of_embeddings, self.number_of_patches)
        # self.batch_data = 100
        if self.training:
            if category_data == 'D':
                for i in range(self.number_of_patches):
                    patch_embeddings = embeddings[:, :, i]  # b * c
                    for j in range(b):
                        self.covariances_d[i, :, :] += torch.outer(
                            patch_embeddings[j, :],
                            patch_embeddings[j, :])  # c * c
                    self.means_d[i, :] += patch_embeddings.sum(dim=0)  # c
                self.n_d += b  # number of images
            elif category_data == 'P':
                for i in range(self.number_of_patches):
                    patch_embeddings = embeddings[:, :, i]  # b * c
                    for j in range(b):
                        self.covariances_p[i, :, :] += torch.outer(
                            patch_embeddings[j, :],
                            patch_embeddings[j, :])  # c * c
                    self.means_p[i, :] += patch_embeddings.sum(dim=0)  # c
                self.n_p += b  # number of images
            elif category_data == 'W':
                for i in range(self.number_of_patches):
                    patch_embeddings = embeddings[:, :, i]  # b * c
                    for j in range(b):
                        self.covariances_w[i, :, :] += torch.outer(
                            patch_embeddings[j, :],
                            patch_embeddings[j, :])  # c * c
                    self.means_w[i, :] += patch_embeddings.sum(dim=0)  # c
                self.n_w += b  # number of images
            else:
                pass
        #     if self.n < 1:
        #         self.data = embeddings.numpy()
        #         self.n += 1  # number of batch
        #     elif self.n % self.batch_data != 0:
        #         self.data = np.concatenate((self.data, embeddings.numpy()), axis=0)
        #         self.n += 1  # number of batch
        #     elif self.n % self.batch_data == 0:
        #         self.i += 1
        #         means = torch.zeros((self.number_of_patches, self.num_of_components,  self.number_of_embeddings))
        #         covariances = torch.zeros((self.number_of_patches, self.num_of_components,  self.number_of_embeddings, self.number_of_embeddings))
        #         for i in range(self.number_of_patches):
        #             patch_embeddings = self.data[:, :, i]
        #             pixel_data = patch_embeddings
        #             self.gm = GaussianMixture(n_components=3, random_state=0).fit(pixel_data)
        #             means[i, :] = torch.tensor(self.gm.means_)
        #             covariances[i, :, :] = torch.tensor(self.gm.covariances_)
        #         self.means += Parameter(means.clone().to(self.device), requires_grad=False)
        #         self.covariances += Parameter(covariances.clone().to(self.device), requires_grad=False)
        #         self.data = embeddings.numpy()
        #         self.n += 1
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
        # b, c, h, w = embedding_dimensions
        # dist_list = torch.zeros((b, h, w), device=self.device)
        # patches = embedding.reshape(b, c, w * h)
        # for i in range(b):
        #     result = torch.zeros((self.num_of_components, h, w), device=self.device)
        #     for j in range(self.num_of_components):
        #         current_patch = patches[i].permute(1, 0)
        #
        #         delta = current_patch - self.learned_means[j, :]
        #         temp = torch.matmul(self.learned_inverse_covariances[j, :, :], delta.unsqueeze(-1))
        #         result[j, :, :] = torch.sqrt(torch.matmul(delta.unsqueeze(1), temp)).reshape(h, w)
        #     result_sum = torch.sum(result, dim=[1, 2])
        #     _, min_idx = torch.min(result_sum, dim=0)
        #     dist_list[i] = result[min_idx, :, :]
        # return dist_list

        b, c, h, w = embedding_dimensions
        result = torch.zeros((b, self.num_of_components, h, w))
        dist = torch.zeros((b, h, w))
        for i in range(self.num_of_components):
            learned_means = self.learned_means[:, i, :]
            learned_inverse_covariances = self.learned_inverse_covariances[:, i, :, :]
            delta = embedding.transpose(2, 1) - learned_means.unsqueeze(0)

            # Calculates the mahalanobis distance in a batched manner
            batched_tensor_result = torch.sqrt(
                torch.einsum('bij,ijk,bik->bi', delta, learned_inverse_covariances, delta)
            )
            result[:, i, :, :] = batched_tensor_result.view((b, h, w))
        result_sum = torch.sum(result, dim=[2,3])
        _, min_idx = torch.min(result_sum, dim=1)

        for j in range(b):
            dist[j, :, :] = result[j, min_idx[j], :, :]
        return dist

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