import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.nn import Parameter

from models import PaDiMBase
from utils.utils import get_embedding


class PaDiMShared(PaDiMBase):

    def __init__(self, params: dict, device):
        super().__init__(params, device)

        self.means = Parameter(
            torch.zeros((self.number_of_embeddings, ), device=self.device),
            requires_grad=False
        )

        self.covariances = Parameter(
            torch.zeros((self.number_of_embeddings, self.number_of_embeddings), device=self.device),
            requires_grad=False
        )

        self.learned_means = Parameter(
            torch.zeros_like(self.means, device=self.device),
            requires_grad=False
        )

        self.learned_covariances = Parameter(
            torch.zeros_like(self.covariances, device=self.device),
            requires_grad=False
        )

        self.learned_inverse_covariances = Parameter(
            torch.zeros_like(self.covariances, device=self.device),
            requires_grad=False
        )

    def calculate_means_and_covariances(self):
        self.learned_means = Parameter(self.means.clone().to(self.device), requires_grad=False)
        self.learned_covariances = Parameter(self.covariances.clone().to(self.device), requires_grad=False)

        epsilon = 0.01

        identity = torch.eye(self.number_of_embeddings, device=self.device)
        self.learned_means /= self.n

        self.learned_covariances -= self.n * torch.outer(self.learned_means, self.learned_means)
        self.learned_covariances /= self.n - 1  # corrected covariance
        self.learned_covariances += epsilon * identity

        self.learned_inverse_covariances = Parameter(torch.linalg.inv(self.learned_covariances), requires_grad=False)

        # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
        #  variables to track the batches)

    def forward(self, x, min_max_norm: bool = True):
        x = x.to(self.device)

        with torch.no_grad():
            features_1, features_2, features_3 = self.backbone(x)

        embeddings = get_embedding(features_1, features_2, features_3, self.embedding_ids, self.device)
        b, c, h, w = embeddings.size()

        if self.training:
            patches = embeddings.permute((0, 2, 3, 1)).reshape((-1, c))

            n_patches = patches.size(0)
            for i in range(n_patches):
                patch = patches[i]
                self.covariances += torch.outer(patch, patch)  # c * c
            self.means += patches.sum(dim=0)
            self.n += n_patches
        else:
            # embeddings = embeddings.view(-1, self.number_of_embeddings, self.number_of_patches)
            return self.calculate_score_map(embeddings, (b, c, h, w), min_max_norm=min_max_norm)

    def _calculate_dist_list(self, embedding, embedding_dimensions: tuple):
        b, c, h, w = embedding_dimensions

        dist_list = torch.zeros((b, h, w), device=self.device)

        patches = embedding.reshape(b, c, w * h)

        for i in range(b):
            current_patch = patches[i].permute(1, 0)

            delta = current_patch - self.learned_means
            temp = torch.matmul(self.learned_inverse_covariances, delta.unsqueeze(-1))
            dist_list[i] = torch.sqrt(torch.matmul(delta.unsqueeze(1), temp)).reshape(h, w)

        return dist_list

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
