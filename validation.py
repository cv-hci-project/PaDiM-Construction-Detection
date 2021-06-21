import argparse
import os

import matplotlib
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from models import PaDiM
from utils.dataloader_utils import get_dataloader, get_device
from utils.utils import (transforms_for_pretrained, get_roc_plot_and_threshold, denormalization_for_pretrained,
                         create_mask)


def create_img_subplot(img, img_score, threshold, vmin, vmax):
    img = denormalization_for_pretrained(img)
    # gt = gts[i].transpose(1, 2, 0).squeeze()
    # heat_map = scores[i] * 255
    heat_map = np.copy(img_score)

    mask = create_mask(img_score, threshold)

    vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
    fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
    fig_img.subplots_adjust(right=0.9)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    for ax_i in ax_img:
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)
    ax_img[0].imshow(img)
    ax_img[0].title.set_text('Image')
    ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
    ax_img[1].imshow(img, cmap='gray', interpolation='none')
    ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    ax_img[1].title.set_text('Predicted heat map')
    ax_img[2].imshow(mask, cmap='gray')
    ax_img[2].title.set_text('Predicted mask')
    ax_img[3].imshow(vis_img)
    ax_img[3].title.set_text('Segmentation result')
    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig_img.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 8,
    }
    cb.set_label('Anomaly Score', fontdict=font)

    return fig_img, ax_img


def save_plot_figs(batch, batch_id, label, scores, threshold, v_max, v_min, path):
    figures = []
    num = len(scores)
    # vmax = scores.max() * 255.
    # vmin = scores.min() * 255.
    vmax = v_max
    vmin = v_min
    for i in range(num):
        classified_as = scores[i].max() > threshold

        fig_img, ax_img = create_img_subplot(batch[i], scores[i], threshold=threshold, vmin=vmin,
                                             vmax=vmax)
        name = "Validation_{}_Image_Classified_as_{}_{}.png".format(int(label[i]), classified_as, i + batch_id * num)
        fig_img.savefig(os.path.join(path, name), dpi=100)


def main():
    parser = argparse.ArgumentParser(description='Validate a PaDiM model')
    parser.add_argument('--load', '-l',
                        dest="experiment_dir",
                        metavar='EXP_DIR',
                        help='Path to the experiment folder, containing a trained PaDiM model')
    parser.add_argument('--validation_config', '-vc',
                        dest="validation_config",
                        metavar='VAL_CFG',
                        help='Path to a validation config to overwrite some parameters of the original experiment',
                        default='configurations/validation.yaml')

    args = parser.parse_args()

    with open(os.path.join(args.experiment_dir, "configuration.yaml"), 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    with open(args.validation_config, 'r') as file:
        try:
            validation_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    image_savepath = os.path.join(args.experiment_dir, "validation")
    os.makedirs(image_savepath, exist_ok=True)

    gpu_id = validation_config["trainer_params"]["gpu"]
    device = get_device(gpu_id)

    print("Device in use: {}".format(device))

    padim = PaDiM(params=config["exp_params"], device=device)
    padim.load_state_dict(torch.load(os.path.join(args.experiment_dir, "padim.pt")))

    # Important to set the model to eval mode, so that in the forward pass of the model the score maps are calculated
    padim.eval()

    crop_size = validation_config["exp_params"]["crop_size"]
    batch_count = validation_config["exp_params"]["batch_count"]
    batch_size = validation_config["exp_params"]["batch_size"]

    config["exp_params"]["batch_size"] = batch_size

    transform = transforms_for_pretrained(crop_size=crop_size)
    normal_data_dataloader = get_dataloader(config["exp_params"], train_split=False, abnormal_data=False,
                                            transform=transform)
    abnormal_data_dataloader = get_dataloader(config["exp_params"], train_split=False, abnormal_data=True,
                                              transform=transform)

    try:
        assert batch_count < len(normal_data_dataloader) and batch_count < len(abnormal_data_dataloader)
    except AssertionError:
        print("Chosen batch count '{}' is larger than there are available batches for the".format(batch_count) +
              " validation sets.")
        raise

    normal_data_iterator = iter(normal_data_dataloader)
    abnormal_data_iterator = iter(abnormal_data_dataloader)

    gt_n_tensor = torch.zeros((batch_count, batch_size, 1))
    gt_a_tensor = torch.ones((batch_count, batch_size, 1))
    scores_n_tensor = torch.zeros((batch_count, batch_size, crop_size, crop_size))
    scores_a_tensor = torch.zeros((batch_count, batch_size, crop_size, crop_size))
    batch_normal = torch.zeros((batch_count, batch_size, 3, crop_size, crop_size))
    batch_abnormal = torch.zeros((batch_count, batch_size, 3, crop_size, crop_size))
    # calculate score map
    for i in tqdm(range(batch_count)):
        batch_n = next(normal_data_iterator)[0]
        batch_a = next(abnormal_data_iterator)[0]
        batch_normal[i] = batch_n
        batch_abnormal[i] = batch_a
        scores_n_tensor[i] = padim(batch_n)
        scores_a_tensor[i] = padim(batch_a)

    scores_all = torch.cat([scores_n_tensor, scores_a_tensor], 0)
    bn, bz, cs1, cs2 = scores_all.shape
    scores_all = scores_all.reshape(bn * bz, cs1, cs2)

    gt_all = torch.cat([gt_n_tensor, gt_a_tensor], 0)
    gt_all = gt_all.reshape(bn * bz, 1)

    v_max = scores_all.max()
    v_min = scores_all.min()

    # calculate metrics
    (fig, _), best_threshold = get_roc_plot_and_threshold(scores_all, gt_all)
    fig.savefig(os.path.join(image_savepath, 'roc_curve.png'), dpi=100)
    print("Saved ROC to {}".format(image_savepath))

    for i in tqdm(range(batch_count)):
        scores_n = scores_n_tensor[i]
        gt_n = gt_n_tensor[i]
        batch_n = batch_normal[i]
        save_plot_figs(batch_n, i, gt_n, scores_n, best_threshold, v_max, v_min, image_savepath)

        scores_a = scores_a_tensor[i]
        gt_a = gt_a_tensor[i]
        batch_a = batch_abnormal[i]
        save_plot_figs(batch_a, i, gt_a, scores_a, best_threshold, v_max, v_min, image_savepath)
    print("Saved validation images to {}".format(image_savepath))


if __name__ == "__main__":
    main()
