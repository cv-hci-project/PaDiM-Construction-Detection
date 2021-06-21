import torch
from torch.utils.data import DataLoader

from datasets import ConcreteCracksDataset, SDNet2018


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
