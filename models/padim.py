import torch
from torch.nn import Module, Parameter

from backbones import backbone_models
from utils.utils import get_embedding, calculate_score_map


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

        self.embedding_ids = torch.randperm(self.backbone.embeddings_size)[:self.number_of_embeddings].to(self.device)

        self.n = 0

        self.learned_means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device),
            requires_grad=False
        )

        self.learned_covariances = Parameter(
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

        # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
        #  variables to track the batches)

    def forward(self, x):
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
            # TODO calculate distance map here
            scores = calculate_score_map(embeddings, (b, c, h, w), self.learned_means,
                                         self.learned_covariances, self.crop_size, min_max_norm=True)
            return torch.Tensor(scores)
