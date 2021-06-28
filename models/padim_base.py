import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.nn import Module, Parameter

from backbones import backbone_models


class PaDiMBase(Module):

    def __init__(self, params: dict, backbone_params: dict,  device):
        super().__init__()

        self.device = device

        self.params = params

        self.backbone = backbone_models[backbone_params["backbone"]](**backbone_params)

        if backbone_params["backbone"] == "vanilla_vae":
            state_dict = torch.load(backbone_params["pretrained_file_path"], map_location=self.device)["state_dict"]
            self.backbone.load_state_dict(state_dict, strict=False)

        self.backbone.to(device)
        self.backbone.eval()

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

    def _calculate_dist_list(self, embedding, embedding_dimensions: tuple):
        raise NotImplementedError()

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

        return torch.Tensor(scores, device=self.device)
