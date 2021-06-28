from typing import List

import torch
from torch import Tensor
from torch import nn


class VanillaVAE(nn.Module):

    embeddings_size = 448
    number_of_patches = 56 * 56  # For a crop_size of 224

    def __init__(self,
                 pretrained_file_path: str,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List,
                 device: torch.device) -> None:
        super().__init__()
        # super().__init__(params)

        self.latent_dim = latent_dim
        self.device = device

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        _in_channels = in_channels

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(_in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            _in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        # TODO I hacked this hardcoded value into here because original author did the same, should be better done
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        # TODO Same as above, redo the hardcoded stuff
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self.eval()

        self.load_state_dict(torch.load(pretrained_file_path, map_location=self.device))

    def forward(self, x: Tensor):

        feature_1 = self.encoder[0](x)
        feature_2 = self.encoder[1](feature_1)
        feature_3 = self.encoder[2](feature_2)

        return feature_1, feature_2, feature_3
