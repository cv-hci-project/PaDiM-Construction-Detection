import torch
from torch.nn import Module, Parameter

from backbones import backbone_models
from utils.utils import get_embedding


class PaDiM(Module):

    def __init__(self, backbone_architecture: str, number_of_embeddings: int, device):
        super().__init__()

        self.device = device

        self.backbone = backbone_models[backbone_architecture]()
        self.backbone.to(device)

        self.number_of_patches = self.backbone.num_patches
        self.number_of_embeddings = number_of_embeddings

        self.means = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings)).to(self.device)
        )
        self.covariances = Parameter(
            torch.zeros((self.number_of_patches, self.number_of_embeddings, self.number_of_embeddings)).to(self.device)
        )

        self.embedding_ids = torch.randperm(self.backbone.embeddings_size)[:number_of_embeddings].to(self.device)

        self.n = 0

        self.calculated_means = None
        self.calculated_covariances = None

    def calculate_means_and_covariances(self):
        self.calculated_means = self.means.clone()
        self.calculated_covariances = self.covariances.clone()

        epsilon = 0.01

        identity = torch.eye(self.number_of_embeddings).to(self.device)
        self.calculated_means /= self.n

        for i in range(self.number_of_patches):
            self.calculated_covariances[i, :, :] -= self.n * torch.outer(self.calculated_means[i, :],
                                                                                 self.calculated_means[i, :])
            self.calculated_covariances[i, :, :] /= self.n - 1  # corrected covariance
            self.calculated_covariances[i, :, :] += epsilon * identity  # constant term

        # TODO we could delete self.means and self.covariances as they are not needed anymore (they are only running
        #  variables to track the batches)

    def forward(self, x):
        # TODO inference step here, i.e. calculating distsance function
        x = x.to(self.device)

        if self.training:
            with torch.no_grad():
                features_1, features_2, features_3 = self.backbone(x)

            embeddings = get_embedding(features_1, features_2, features_3, self.embedding_ids)
            b, c, h, w = embeddings.size()

            embeddings = embeddings.view(-1, self.number_of_embeddings, self.number_of_patches)

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
            raise NotImplementedError()
