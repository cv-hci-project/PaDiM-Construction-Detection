from typing import Tuple

from torch import Tensor
from torch.nn import Module
import torch
from efficientnet_pytorch import EfficientNet


class EffNet(Module):

    embeddings_size = 344
    number_of_patches = 64 * 64  # For a crop_size of 224

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        self.efficientnet.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the three intermediary layers from the WideResNet50
        pre-trained model.
        Params
        ======
            x: Tensor - the input tensor of size (b * c * w * h)
        Returns
        =======
            feature_1: Tensor - the residual from layer 1
            feature_2: Tensor - the residual from layer 2
            feature_3: Tensor - the residual from layer 3
        """

        x = self.efficientnet._swish(self.efficientnet._bn0(self.efficientnet._conv_stem(x)))

        results = []
        layer_indices = [7, 19, 26]
        max_idx = max(layer_indices)
        for i, block in enumerate(self.efficientnet._blocks):
            x = block(x)
            if i in layer_indices:
                results.append(x)
            if i > max_idx:
               break

        return results[0], results[1], results[2]
