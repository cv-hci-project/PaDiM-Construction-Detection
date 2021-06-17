import argparse
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import Tensor
from tqdm import tqdm
import yaml

from backbones import backbone_models
from utils.dataloader_utils import get_dataloader
from utils.utils import transforms_for_pretrained, get_embedding


def setup_padim(backbone: torch.nn.Module, number_of_embeddings, device) -> Tuple[Tensor, Tensor, Tensor]:
    means = torch.zeros((backbone.num_patches, number_of_embeddings)).to(device)
    covariances = torch.zeros((backbone.num_patches, number_of_embeddings, number_of_embeddings)).to(device)

    embedding_ids = torch.randperm(backbone.embeddings_size)[:number_of_embeddings].to(device)

    return means, covariances, embedding_ids


def main():
    parser = argparse.ArgumentParser(description='Train a PaDiM model')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='Path to the configuration file',
                        default='configurations/default.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    gpu_id = config["trainer_params"]["gpu"]

    if gpu_id >= 0:
        device = torch.device('cuda:{}'.format(gpu_id))

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Chose gpu_id '{}', but no GPU is available. If you want to use the CPU, set it to '-1'".format(gpu_id))
    else:
        device = torch.device('cpu')

    print("Device in use: {}".format(device))

    backbone = backbone_models[config['exp_params']['backbone']]()
    backbone.to(device)

    number_of_embeddings = config['exp_params']["number_of_embeddings"]
    number_of_patches = backbone.num_patches

    means, covariances, embedding_ids = setup_padim(backbone, number_of_embeddings, device)
    n = 0

    transform = transforms_for_pretrained(crop_size=config["exp_params"]["crop_size"])

    normal_data_dataloader = get_dataloader(config['exp_params'], train_split=True, abnormal_data=False, shuffle=True,
                                            transform=transform)

    for batch_id, (batch, _) in tqdm(enumerate(normal_data_dataloader), total=len(normal_data_dataloader)):

        batch = batch.to(device)

        with torch.no_grad():
            features_1, features_2, features_3 = backbone(batch)

        embeddings = get_embedding(features_1, features_2, features_3, embedding_ids)
        B, C, H, W = embeddings.size()

        embeddings = embeddings.view(-1, number_of_embeddings, number_of_patches)

        for i in range(number_of_patches):
            patch_embeddings = embeddings[:, :, i]  # b * c
            for j in range(B):
                covariances[i, :, :] += torch.outer(
                    patch_embeddings[j, :],
                    patch_embeddings[j, :])  # c * c
            means[i, :] += patch_embeddings.sum(dim=0)  # c
        n += B  # number of images

    _means = means.clone()
    _covs = covariances.clone()

    epsilon = 0.01

    identity = torch.eye(number_of_embeddings).to(device)
    _means /= n

    for i in range(number_of_patches):
        _covs[i, :, :] -= n * torch.outer(_means[i, :], _means[i, :])
        _covs[i, :, :] /= n - 1  # corrected covariance
        _covs[i, :, :] += epsilon * identity  # constant term

    # self.means = torch.nn.Parameter(means, requires_grad=False).to(self.device)
    # self.covs = torch.nn.Parameter(covs, requires_grad=False).to(self.device)
    # self.covs_inv = torch.nn.Parameter(torch.from_numpy(np.linalg.inv(self.covs.cpu().numpy())),
    #                                    requires_grad=False).to(self.device)


if __name__ == "__main__":
    main()
