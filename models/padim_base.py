import torch
from torch.nn import Module, Parameter

from backbones import backbone_models


class PaDiMBase(Module):

    def __init__(self, params: dict, device):
        super().__init__()

        self.device = device

        self.params = params

        self.backbone = backbone_models[params["backbone"]]()
        self.backbone.to(device)

        self.number_of_patches = self.backbone.number_of_patches
        self.number_of_embeddings = params["number_of_embeddings"]

        self.crop_size = params["crop_size"]

        self.n = 0

        self.embedding_ids = Parameter(
            torch.randperm(self.backbone.embeddings_size, device=self.device)[:self.number_of_embeddings],
            requires_grad=False
        )

    def calculate_means_and_covariances(self):
        raise NotImplementedError()

    def forward(self, x, min_max_norm: bool = True):
        raise NotImplementedError()

    def calculate_score_map(self, embedding, embedding_dimensions: tuple, min_max_norm: bool) -> torch.Tensor:
        raise NotImplementedError()
