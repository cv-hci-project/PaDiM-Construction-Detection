import argparse
import os

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from test_tube import Experiment
from torch import nn
from torch import optim
from torch.backends import cudnn
from torchvision.models import resnet18
from torchvision.transforms import transforms
from tqdm import tqdm

from utils.dataloader_utils import get_dataloader, get_device


def iterate_dataset(dataloader, model, device, optimizer, criterion, experiment):
    data_iterator = iter(dataloader)

    running_loss = 0.0

    for batch in tqdm(data_iterator):
        x, labels = batch[0].to(device), batch[1].float().to(device)

        optimizer.zero_grad()

        outputs = model(x).flatten()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        experiment.log({"loss": loss.item()})


def validate(dataloader, model, device):
    data_iterator = iter(dataloader)

    all_predictions = []
    true_labels = []

    pbar = tqdm(data_iterator, desc="Validation")
    for batch in pbar:
        x, labels = batch[0].to(device), batch[1].cpu().numpy()

        outputs = model(x).flatten()

        predictions = (outputs > 0.5).int().cpu().numpy()

        all_predictions.extend(predictions)
        true_labels.extend(labels)

    return all_predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description='Train a supervised model as a baseline for PaDiM')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='Path to the configuration file')

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

    print("Started experiment version {}".format(experiment.version))

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

    # Get the data
    set_range = transforms.Lambda(lambda x: 2.0 * x - 1.0)
    resize = config["exp_params"]["resize"]
    transformations = transforms.Compose([transforms.Resize((resize, resize)),
                                          transforms.ToTensor(),
                                          set_range])

    normal_train_data_dataloader = get_dataloader(config['exp_params'], train_split=True, abnormal_data=False, shuffle=True,
                                            transform=transformations)

    normal_val_data_dataloader = get_dataloader(config['exp_params'], train_split=False, abnormal_data=False, shuffle=True,
                                            transform=transformations)

    abnormal_train_data_dataloader = get_dataloader(config['exp_params'], train_split=True, abnormal_data=True, shuffle=True,
                                              transform=transformations)

    abnormal_val_data_dataloader = get_dataloader(config['exp_params'], train_split=False, abnormal_data=True, shuffle=True,
                                              transform=transformations)

    max_epochs = config["trainer_params"]["max_epochs"]

    # Get the model
    if config["exp_params"]["architecture"] == "resnet18":
        model = resnet18(pretrained=False)
        model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    else:
        raise RuntimeError("Chosen model architecture '{}' is unknown, choose another.".format(
            config["exp_params"]["architecture"])
        )

    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    pbar = tqdm(range(max_epochs), desc="Epoch")
    for epoch in pbar:
        pbar.set_description("Epoch {}".format(epoch))
        iterate_dataset(normal_train_data_dataloader, model, device, optimizer, criterion, experiment)
        iterate_dataset(abnormal_train_data_dataloader, model, device, optimizer, criterion, experiment)

        torch.save(model.state_dict(), os.path.join(log_dir, "baseline.pt"))

    model.train(False)

    predictions_normal, true_labels_normal = validate(normal_val_data_dataloader, model, device)
    predictions_abnormal, true_labels_abnormal = validate(abnormal_val_data_dataloader, model, device)

    all_predictions = predictions_normal + predictions_abnormal
    all_true_labels = true_labels_normal + true_labels_abnormal

    fpr, tpr, thresholds = roc_curve(all_true_labels, all_predictions)
    calculated_roc_auc_score = roc_auc_score(all_true_labels, all_predictions)

    fig, ax = plt.subplots(1, 1)
    fig_img_rocauc = ax

    fig_img_rocauc.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(calculated_roc_auc_score))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    experiment.add_figure(tag="val_roc_auc", figure=fig)


if __name__ == "__main__":
    main()
