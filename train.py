import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from test_tube import Experiment
from tqdm import tqdm

from models import PaDiM
from utils.dataloader_utils import get_dataloader, get_device
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

    experiment = Experiment(
        save_dir=config['logging_params']['save_dir'],
        name=config['logging_params']['name'],
        debug=False,
        create_git_tag=False
    )

    # Save the config
    log_dir = experiment.get_logdir().split("tf")[0]
    with open(os.path.join(log_dir, "configuration.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    cudnn.deterministic = True
    cudnn.benchmark = False

    gpu_id = config["trainer_params"]["gpu"]
    device = get_device(gpu_id)

    print("Device in use: {}".format(device))

    padim = PaDiM(params=config["exp_params"], device=device)

    transform = transforms_for_pretrained(crop_size=config["exp_params"]["crop_size"])

    normal_data_dataloader = get_dataloader(config['exp_params'], train_split=True, abnormal_data=False, shuffle=True,
                                            transform=transform)

    for batch_id, (batch, _) in tqdm(enumerate(normal_data_dataloader), total=len(normal_data_dataloader)):
        padim(batch)

    padim.calculate_means_and_covariances()

    # Saves the trained model inside the logging directory
    torch.save(padim.state_dict(), os.path.join(log_dir, "padim.pt"))


if __name__ == "__main__":
    main()
