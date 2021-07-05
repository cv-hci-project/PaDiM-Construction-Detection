import argparse
import os
import shutil

import matplotlib
import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from tqdm import tqdm

from backbones import backbone_kinds
from models import registered_padim_models
from utils.dataloader_utils import get_dataloader, get_device, get_transformations, denormalize_batch
from utils.utils import get_roc_plot_and_threshold, create_mask


def save_plot_figs(batch, batch_id, label, scores, threshold, v_max, v_min, path, backbone_kind):
    figures = []
    num = len(scores)
    # vmax = scores.max() * 255.
    # vmin = scores.min() * 255.
    vmax = v_max
    vmin = v_min
    for i in range(num):
        classified_as = scores[i].max() > threshold

        fig_img, ax_img = create_img_subplot(batch[i], scores[i], threshold=threshold, vmin=vmin,
                                             vmax=vmax, backbone_kind=backbone_kind)
        name = "Validation_{}_Image_Classified_as_{}_{}.png".format(int(label[i]), classified_as, i + batch_id * num)
        save_path = os.path.join(path, 'Classified_as_{}'.format(classified_as))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig_img.savefig(os.path.join(save_path, name), dpi=100)


def create_img_subplot(img, img_score, threshold, vmin, vmax, backbone_kind):
    img = denormalize_batch(backbone_kind=backbone_kind, x=img.cpu().numpy())
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


def save_grid_plot(batch, batch_id, label, scores, threshold, v_max, v_min, path, backbone_kind):
    # figures = []
    # num = len(scores)
    # vmax = scores.max() * 255.
    # vmin = scores.min() * 255.
    vmax = v_max
    vmin = v_min

    # TODO: separate in predicted positive/negative rather than GT positive/negative?
    # classified_as = scores.max() > threshold

    img_plot, _ = create_grid_plot(batch, scores, threshold=threshold, vmin=vmin, vmax=vmax, backbone_kind=backbone_kind)
    img_plot.set_size_inches(20, 12)
    img_plot.savefig(os.path.join(path, "Validation_Batch_{}_{}".format(batch_id, "Positive" if label[0] == True else "Negative")), dpi=100)


def create_grid_plot(imgs, img_scores, threshold, vmin, vmax, backbone_kind):
    original = [None] * imgs.shape[0]
    heat_maps = [None] * imgs.shape[0]
    masks = [None] * imgs.shape[0]
    vis_imgs = [None] * imgs.shape[0]

    for i in range(imgs.shape[0]):
        if isinstance(threshold, list):
            _threshold = threshold[i]
        else:
            _threshold = threshold
        original[i] = denormalize_batch(backbone_kind=backbone_kind, x=imgs[i].cpu().numpy())
        heat_maps[i] = np.copy(img_scores[i])
        masks[i] = create_mask(img_scores[i], _threshold)
        vis_imgs[i] = mark_boundaries(original[i], masks[i], color=(1, 0, 0), mode='thick')

    # Transform to tensor and prepare for plotting: BxCxWxH
    original = torch.Tensor(original).permute(0, 3, 1, 2)
    heat_maps = torch.Tensor(heat_maps).unsqueeze(1)
    masks = torch.Tensor(masks).unsqueeze(1)
    vis_imgs = torch.Tensor(vis_imgs).permute(0, 3, 1, 2)

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(nrows=2, ncols=2)
    for ax_i in (ax11, ax12, ax21, ax22):
        ax_i.axes.xaxis.set_visible(False)
        ax_i.axes.yaxis.set_visible(False)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    ax11.imshow(vutils.make_grid(original, normalize=True, nrow=8).permute(1,2,0).numpy())
    ax11.set_title("Original")

    # TODO: fix heat maps
    ax = ax12.imshow(vutils.make_grid(heat_maps, normalize=True, nrow=8).permute(1,2,0).numpy(), cmap='jet', norm=norm)
    ax12.imshow(vutils.make_grid(original, normalize=True, nrow=8).permute(1,2,0).numpy(), cmap='gray', interpolation='none')
    ax12.imshow(vutils.make_grid(heat_maps, normalize=True, nrow=8).permute(1,2,0).numpy(),
               cmap='jet', alpha=0.5, interpolation='none')
    ax12.set_title("Predicted heat maps")

    ax21.imshow(vutils.make_grid(masks, normalize=True, nrow=8).permute(1,2,0).numpy(), cmap='gray')
    ax21.set_title("Predicted masks")

    ax22.imshow(vutils.make_grid(vis_imgs, normalize=True, nrow=8).permute(1,2,0).numpy())
    ax22.set_title("Segmentation results")

    left = 0.92
    bottom = 0.15
    width = 0.015
    height = 1 - 2 * bottom
    rect = [left, bottom, width, height]
    cbar_ax = fig.add_axes(rect)
    cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
    cb.ax.tick_params(labelsize=8)
    font = {
        'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 12,
    }
    cb.set_label('Anomaly Score', fontdict=font)

    return fig, (ax11, ax12, ax21, ax22)


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

    gpu_id = validation_config["trainer_params"]["gpu"]
    device = get_device(gpu_id)

    print("Device in use: {}".format(device))

    # Use this to replace the data paths, for example when training on the server but validating locally
    # config["exp_params"]["dataset"] = "SDNET2018"
    # config["exp_params"]["data_path"] = "/home/pdeubel/PycharmProjects/data/SDNET2018"
    # config["exp_params"]["data_path"] = "/home/pdeubel/PycharmProjects/data/Concrete-Crack-Images"

    padim = registered_padim_models[config["exp_params"]["padim_mode"]](params=config["exp_params"],
                                                                        backbone_params=config["backbone_params"],
                                                                        device=device)
    padim.load_state_dict(torch.load(os.path.join(args.experiment_dir, "padim.pt"), map_location=device))

    # Important to set the model to eval mode, so that in the forward pass of the model the score maps are calculated
    padim.eval()

    crop_size = validation_config["exp_params"]["crop_size"]
    batch_size = validation_config["exp_params"]["batch_size"]

    config["exp_params"]["batch_size"] = batch_size
    config["exp_params"]["dataloader_workers"] = 0

    backbone_kind = backbone_kinds[config["backbone_params"]["backbone"]]
    min_max_normalization = validation_config["exp_params"]["min_max_normalization"]

    transform = get_transformations(backbone_kind=backbone_kind, crop_size=crop_size)
    normal_data_dataloader = get_dataloader(config["exp_params"], train_split=False, abnormal_data=False,
                                            transform=transform)
    abnormal_data_dataloader = get_dataloader(config["exp_params"], train_split=False, abnormal_data=True,
                                              transform=transform)

    try:
        batch_count = validation_config["exp_params"]["batch_count"]
    except KeyError:
        batch_count = min(len(normal_data_dataloader), len(abnormal_data_dataloader))

    try:
        assert batch_count <= len(normal_data_dataloader) and batch_count <= len(abnormal_data_dataloader)
    except AssertionError:
        print("Chosen batch count '{}' is larger than there are available batches for the".format(batch_count) +
              " validation sets.")
        raise

    number_visualization_batches = validation_config["exp_params"]["number_visualization_batches"]

    normal_data_iterator = iter(normal_data_dataloader)
    abnormal_data_iterator = iter(abnormal_data_dataloader)

    gt_n_tensor = torch.zeros((batch_count, batch_size, 1))
    gt_a_tensor = torch.ones((batch_count, batch_size, 1))

    predictions_n = []
    predictions_a = []

    predictions_n_per_category = {}
    predictions_a_per_category = {}

    visualization_batches_n = []
    visualization_batches_a = []

    scores_n_tensor = torch.zeros((number_visualization_batches, batch_size, crop_size, crop_size))
    scores_a_tensor = torch.zeros((number_visualization_batches, batch_size, crop_size, crop_size))
    batch_normal = torch.zeros((number_visualization_batches, batch_size, 3, crop_size, crop_size))
    batch_abnormal = torch.zeros((number_visualization_batches, batch_size, 3, crop_size, crop_size))

    # calculate score map
    for i in tqdm(range(batch_count)):
        # batch_n = next(normal_data_iterator)[0]
        # batch_a = next(abnormal_data_iterator)[0]

        batch_n = next(normal_data_iterator)
        batch_a = next(abnormal_data_iterator)

        _score_n = padim(batch_n, min_max_norm=min_max_normalization)
        _score_a = padim(batch_a, min_max_norm=min_max_normalization)

        assert _score_n.size() == _score_a.size()

        if len(_score_n.size()) == 4:
            for category in range(_score_n.size(0)):
                try:
                    predictions_n_per_category[category]
                except KeyError:
                    predictions_n_per_category[category] = []

                try:
                    predictions_a_per_category[category]
                except KeyError:
                    predictions_a_per_category[category] = []

                predictions_n_per_category[category].extend(_score_n[category].reshape(_score_n.size(1), -1).max(axis=1)[0].cpu().numpy())
                predictions_a_per_category[category].extend(_score_a[category].reshape(_score_a.size(1), -1).max(axis=1)[0].cpu().numpy())
        else:

            predictions_n.append(_score_n.reshape(_score_n.shape[0], -1).max(axis=1)[0].cpu().numpy())
            predictions_a.append(_score_a.reshape(_score_a.shape[0], -1).max(axis=1)[0].cpu().numpy())

        if i < number_visualization_batches:
            batch_normal[i] = batch_n[0]
            batch_abnormal[i] = batch_a[0]

            if len(_score_n.size()) == 4:
                visualization_batches_n.append(_score_n)
                visualization_batches_a.append(_score_a)
            else:
                scores_n_tensor[i] = _score_n
                scores_a_tensor[i] = _score_a

    for k, v in predictions_a_per_category.items():
        predictions_n_per_category[k].extend(v)

    # predictions_n = np.array(predictions_n).flatten()
    # predictions_a = np.array(predictions_a).flatten()

    # predictions = np.concatenate([predictions_n, predictions_a])

    gt_all = torch.cat([gt_n_tensor, gt_a_tensor], 0)
    gt_all = gt_all.reshape(-1, 1)


    image_savepath = os.path.join(args.experiment_dir, "validation")

    # Delete entries of the validation folder if it exists so that there is no overlap between validations
    try:
        shutil.rmtree(image_savepath)
    except FileNotFoundError:
        # Already deleted, pass
        pass

    os.makedirs(image_savepath, exist_ok=True)

    # calculate metrics
    (fig, _), best_threshold, predicted_category, category_thresholds = get_roc_plot_and_threshold(predictions=predictions_n_per_category, gt_list=gt_all)
    fig.savefig(os.path.join(image_savepath, 'roc_curve.png'), dpi=100)
    print("Saved ROC to {}".format(image_savepath))

    if predicted_category is not None:
        v_max = float(max([x.max() for x in visualization_batches_n + visualization_batches_a]))
        v_min = float(min([x.min() for x in visualization_batches_n + visualization_batches_a]))

        current_index_n = 0
        current_index_a = len(predicted_category) // 2
        for i in tqdm(range(len(visualization_batches_n))):
            scores_n = visualization_batches_n[i]
            scores_a = visualization_batches_a[i]
            new_scores_n = []
            new_scores_a = []

            chosen_thresholds_n = []
            chosen_thresholds_a = []

            predicted_category_n = predicted_category[current_index_n:current_index_n + batch_size]
            predicted_category_a = predicted_category[current_index_a:current_index_a + batch_size]

            for j in range(scores_n.size(1)):
                new_scores_n.append(scores_n[predicted_category_n[j], j])
                new_scores_a.append(scores_a[predicted_category_a[j], j])

                chosen_thresholds_n.append(category_thresholds[predicted_category_n[j]])
                chosen_thresholds_a.append(category_thresholds[predicted_category_a[j]])

            gt_n = gt_n_tensor[i]
            batch_n = batch_normal[i]
            save_grid_plot(batch_n, i, gt_n, np.array(new_scores_n, dtype=object), chosen_thresholds_n, v_max, v_min, image_savepath,
                           backbone_kind=backbone_kind)

            gt_a = gt_a_tensor[i]
            batch_a = batch_abnormal[i]
            save_grid_plot(batch_a, i, gt_a, np.array(new_scores_a, dtype=object), chosen_thresholds_a, v_max, v_min, image_savepath,
                           backbone_kind=backbone_kind)

            current_index_n += batch_size
            current_index_a += batch_size
    else:
        scores_all = torch.cat([scores_n_tensor, scores_a_tensor], 0)
        bn, bz, cs1, cs2 = scores_all.shape
        scores_all = scores_all.reshape(bn * bz, cs1, cs2)

        v_max = scores_all.max()
        v_min = scores_all.min()

        for i in tqdm(range(number_visualization_batches)):
            scores_n = scores_n_tensor[i]
            gt_n = gt_n_tensor[i]
            batch_n = batch_normal[i]
            save_grid_plot(batch_n, i, gt_n, scores_n, best_threshold, v_max, v_min, image_savepath,
                           backbone_kind=backbone_kind)

            scores_a = scores_a_tensor[i]
            gt_a = gt_a_tensor[i]
            batch_a = batch_abnormal[i]
            save_grid_plot(batch_a, i, gt_a, scores_a, best_threshold, v_max, v_min, image_savepath,
                           backbone_kind=backbone_kind)

    print("Saved validation images to {}".format(image_savepath))


if __name__ == "__main__":
    main()
