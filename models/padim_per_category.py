import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.nn import Parameter

from models import PaDiMBase
from utils.utils import get_embedding


class PaDiMPerCategory(PaDiMBase):

    def __init__(self, params: dict, backbone_params: dict, device):
        super().__init__(params=params, backbone_params=backbone_params, device=device)
        self.number_of_categories = self.params["number_of_categories"]

        self.means = []
        self.covariances = []

        self.learned_means = []
        self.learned_covariances = []
        self.learned_inverse_covariances = []

        self.counts = []

        for _ in range(self.number_of_categories):
            self.means.append(Parameter(torch.zeros((self.number_of_patches, self.number_of_embeddings), device=self.device), requires_grad=False))
            self.covariances.append(Parameter(torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings), device=self.device), requires_grad=False))

            self.learned_means.append(Parameter(torch.zeros((self.number_of_patches, self.number_of_embeddings), device=self.device), requires_grad=False))
            self.learned_covariances.append(Parameter(torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings), device=self.device), requires_grad=False))
            self.learned_inverse_covariances.append(Parameter(torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings), device=self.device), requires_grad=False))

            self.counts.append(0)

    def calculate_means_and_covariances(self):
        for category in range(self.number_of_categories):

            self.learned_means[category] = Parameter(self.means[category].clone().to(self.device),
                                                     requires_grad=False)
            self.learned_covariances[category] = Parameter(self.covariances[category].clone().to(self.device),
                                                           requires_grad=False)

            epsilon = 0.01

            identity = torch.eye(self.number_of_embeddings).to(self.device)
            self.learned_means[category] /= self.counts[category]

            for i in range(self.number_of_patches):
                self.learned_covariances[category][i, :, :] -= self.counts[category] * torch.outer(self.learned_means[category][i, :], self.learned_means[category][i, :])
                self.learned_covariances[category][i, :, :] /= self.counts[category] - 1  # corrected covariance
                self.learned_covariances[category][i, :, :] += epsilon * identity  # constant term

            self.learned_inverse_covariances[category] = Parameter(torch.linalg.inv(self.learned_covariances[category]), requires_grad=False)

            # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
            #  variables to track the batches)

    def forward(self, batch, min_max_norm: bool = True):
        x, labels = batch

        x = x.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            features_1, features_2, features_3 = self.backbone(x)

        embeddings = get_embedding(features_1, features_2, features_3, self.embedding_ids, self.device)
        b, c, h, w = embeddings.size()

        embeddings = embeddings.view(-1, self.number_of_embeddings, self.number_of_patches)

        if self.training:
            for i in range(self.number_of_patches):
                patch_embeddings = embeddings[:, :, i]  # b * c
                for j in range(b):
                    self.covariances[labels[j]][i, :, :] += torch.outer(
                        patch_embeddings[j, :],
                        patch_embeddings[j, :]) # c * c
                    self.counts[labels[j]] += 1
                    self.means[labels[j]] += patch_embeddings[j, :]
        else:
            return self.calculate_score_map(embeddings, (b, c, h, w), min_max_norm=min_max_norm)

    def calculate_score_map(self, embedding, embedding_dimensions: tuple, min_max_norm: bool) -> torch.Tensor:
        scores_per_category = []

        for category in range(self.number_of_categories):
            dist_list = self._calculate_dist_list(embedding, embedding_dimensions, category)

            # Upsample
            score_map = F.interpolate(dist_list.unsqueeze(1), size=self.crop_size, mode='bilinear',
                                      align_corners=False).squeeze().cpu().numpy()

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

            scores_per_category.append(scores)

        return torch.Tensor(scores_per_category, device=self.device)

    def _calculate_dist_list(self, embedding, embedding_dimensions: tuple, category: int):
        b, c, h, w = embedding_dimensions

        delta = embedding.transpose(2, 1) - self.learned_means[category].unsqueeze(0)

        # Calculates the mahalanobis distance in a batched manner
        batched_tensor_result = torch.sqrt(
            torch.einsum('bij,ijk,bik->bi', delta, self.learned_inverse_covariances[category], delta)
        )

        return batched_tensor_result.view((b, h, w))
