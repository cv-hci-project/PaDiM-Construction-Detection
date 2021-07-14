import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.nn import Module, Parameter

from backbones import backbone_models, backbone_kinds
from utils.dataloader_utils import get_dataloader, get_transformations


class PaDiMBase(Module):

    def __init__(self, params: dict, backbone_params: dict,  device):
        super().__init__()

        self.device = device

        self.params = params
        self.crop_size = self.params["crop_size"]

        self.backbone = backbone_models[backbone_params["backbone"]](**backbone_params)
        backbone_kind = backbone_kinds[backbone_params["backbone"]]

        if backbone_kind == "vae":
            state_dict = torch.load(backbone_params["pretrained_file_path"], map_location=self.device)["state_dict"]
            self.backbone.load_state_dict(state_dict, strict=False)

        self.backbone.to(device)
        self.backbone.eval()

        transform = get_transformations(backbone_kind=backbone_kind, crop_size=self.crop_size)

        normal_data_dataloader = get_dataloader(self.params, split="train", abnormal_data=False, shuffle=True,
                                                transform=transform)

        test_batch = next(iter(normal_data_dataloader))[0].to(device)
        feature_1, feature_2, feature_3 = self.backbone(test_batch)

        self.number_of_patches = feature_1.size(2) * feature_1.size(3)
        embeddings_size = feature_1.size(1) + feature_2.size(1) + feature_3.size(1)

        if "number_of_embeddings" in self.params:
            self.number_of_embeddings = params["number_of_embeddings"]

            self.embedding_ids = Parameter(
                torch.randperm(embeddings_size, device=self.device)[:self.number_of_embeddings],
                requires_grad=False
            )
        else:
            self.number_of_embeddings = embeddings_size

            self.embedding_ids = Parameter(
                torch.arange(0, self.number_of_embeddings, device=self.device),
                requires_grad=False
            )

        self.n = 0


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

        return torch.tensor(scores, device=self.device)
