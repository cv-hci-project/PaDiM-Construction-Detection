import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import (ConcreteCracksDataset, SDNet2018, SDNet2018CleanedThreshold, SDNet2018CleanedThresholdPercentile,
                      SDNet2018PerCategory, SDNet2018CleanedPercentileAll)


def get_dataloader(params: dict, train_split: bool, abnormal_data: bool = False, shuffle: bool = True, transform=None):
    additional_dataloader_args = {'num_workers': params["dataloader_workers"], 'pin_memory': True}

    split = "train" if train_split else "val"

    if params["dataset"] == 'concrete-cracks':
        dataset = ConcreteCracksDataset(root_dir=params['data_path'],
                                        split=split,
                                        abnormal_data=abnormal_data,
                                        transform=transform)
    elif params["dataset"] == 'SDNET2018':
        dataset = SDNet2018(root_dir=params['data_path'],
                            split=split,
                            abnormal_data=abnormal_data,
                            transform=transform)
    elif params["dataset"] == 'SDNET2018PerCategory':
        dataset = SDNet2018PerCategory(root_dir=params['data_path'],
                                       split=split,
                                       abnormal_data=abnormal_data,
                                       transform=transform)
    elif params["dataset"] == 'SDNet2018CleanedThreshold':
        dataset = SDNet2018CleanedThreshold(root_dir=params['data_path'],
                                            split=split,
                                            abnormal_data=abnormal_data,
                                            transform=transform)
    elif params["dataset"] == 'SDNet2018CleanedThresholdPercentile':
        dataset = SDNet2018CleanedThresholdPercentile(root_dir=params['data_path'],
                                                      split=split,
                                                      abnormal_data=abnormal_data,
                                                      transform=transform)
    elif params["dataset"] == 'SDNet2018CleanedPercentileAll':
        dataset = SDNet2018CleanedPercentileAll(root_dir=params['data_path'],
                                                split=split,
                                                abnormal_data=abnormal_data,
                                                transform=transform)
    else:
        raise ValueError('Undefined dataset type')

    dataloader = DataLoader(dataset,
                            batch_size=params['batch_size'],
                            shuffle=shuffle,
                            drop_last=True,
                            **additional_dataloader_args)

    return dataloader


def get_device(gpu_id: int):
    if gpu_id >= 0:
        device = torch.device('cuda:{}'.format(gpu_id))

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Chose gpu_id '{}', but no GPU is available. If you want to use the CPU, set it to '-1'".format(gpu_id))
    else:
        device = torch.device('cpu')

    return device


def get_transformations(backbone_kind: str, crop_size: int):
    if backbone_kind == "pretrained_imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transformations = transforms.Compose([transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=mean,
                                                                   std=std)])
    elif backbone_kind == "vae":
        set_range = transforms.Lambda(lambda x: 2.0 * x - 1.0)

        transformations = transforms.Compose([transforms.Resize((crop_size, crop_size)),
                                              transforms.ToTensor(),
                                              set_range])
    else:
        raise RuntimeError("Chosen backbone_kind '{}' not supported.".format(backbone_kind))

    return transformations


def denormalize_batch(backbone_kind: str, x: np.ndarray):
    if backbone_kind == "pretrained_imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    elif backbone_kind == "vae":
        x = (x.transpose(1, 2, 0) + 1.0) / 2.0
    else:
        raise RuntimeError("Chosen backbone_kind '{}' not supported.".format(backbone_kind))

    return x
