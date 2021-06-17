import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import yaml

from backbones import backbone_models
from utils.dataloader_utils import get_dataloader
from utils.utils import transforms_for_pretrained


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

    backbone = backbone_models(config['arch'])
    backbone.to(device)

    transform = transforms_for_pretrained(crop_size=config["exp_params"]["crop_size"])

    normal_data_dataloader = get_dataloader(config['exp_params'], train_split=True, abnormal_data=False, shuffle=True,
                                            transform=transform)

    for batch_id, (batch, labels) in enumerate(tqdm(normal_data_dataloader)):
        batch = batch.to(device)
        labels = labels.to(device)



if __name__ == "__main__":
    main()