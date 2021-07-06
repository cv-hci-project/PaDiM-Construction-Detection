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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from backbones import backbone_kinds
from models import registered_padim_models
from utils.dataloader_utils import get_dataloader, get_device, get_transformations, denormalize_batch
from utils.utils import create_mask


def test(predictions):
    category_thresholds = []

    for category, value in predictions.items():
        predictions_per_category = np.asarray(value)
        gt_list = np.asarray(gt_list).flatten()

        # fpr, tpr, thresholds = roc_curve(gt_list, predictions_per_category)
        # category_thresholds.append(thresholds[np.argmax(tpr - fpr)])
        #
        precision, recall, thresholds = precision_recall_curve(gt_list, predictions_per_category)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        category_thresholds.append(thresholds[np.argmax(f1)])

    differences = np.array([v for v in predictions.values()]) - np.expand_dims(category_thresholds, 1)
    predicted_category = np.argmin(differences, axis=0)

    # If an element is larger than 0 that means that the difference between value and threshold was non-negative
    # and therefore the value is above the threshold, meaning it was classified as an anomaly
    all_predictions = []
    for i in range(differences.shape[1]):
        all_predictions.append(differences[predicted_category[i], i])

    calculated_predictions = np.array([x > 0 for x in all_predictions], dtype=int)

    # for i in range(all_predictions.shape[0]):
    #     for j in range(len(category_thresholds)):
    #         if predictions[j][i] <= category_thresholds[j]:
    #             all_predictions[i] = 0
    #             break


def get_roc_plot_and_threshold(predictions, gt_list):
    predictions = np.asarray(predictions)
    gt_list = np.asarray(gt_list)

    fpr, tpr, thresholds = roc_curve(gt_list, predictions)
    img_roc_auc = roc_auc_score(gt_list, predictions)

    fig, ax = plt.subplots(1, 1)
    fig_img_rocauc = ax

    fig_img_rocauc.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(img_roc_auc))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    precision, recall, thresholds = precision_recall_curve(gt_list, predictions)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_threshold = thresholds[np.argmax(f1)]

    return (fig, ax), best_threshold


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


def iterate_through_data(padim, dataloader, _batch_count, _min_max_norm, _number_visualization_batches,
                         abnormal_data: bool):
    data_iterator = iter(dataloader)

    if not abnormal_data:
        ground_truth = 0
    else:
        ground_truth = 1

    predictions = []
    visualization_batches = []
    visualization_score_maps = []

    for i in tqdm(range(_batch_count)):
        batch = next(data_iterator)
        labels = batch[1]

        # Dim: 3 x batch_size x crop_size x crop_size
        score_maps_per_category = padim(batch, min_max_norm=_min_max_norm)

        for j in range(score_maps_per_category.size(1)):
            current_score_map = score_maps_per_category[:, j, :, :]
            anomaly_score = current_score_map.reshape(current_score_map.size(0), -1).max(dim=1)[0]
            predictions.append([anomaly_score, int(labels[j].cpu()), ground_truth])

        if i < _number_visualization_batches:
            visualization_batches.append(batch)
            visualization_score_maps.append(score_maps_per_category)

    return predictions, visualization_batches, visualization_score_maps


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
    parser.add_argument('--min_mode', '-m',
                        dest="min_mode",
                        action="store_true",
                        help='Use min mode to validate')
    parser.add_argument('--real_labels', '-r',
                        dest="real_labels",
                        action="store_true",
                        help='Use real labels mode to validate')
    parser.add_argument('--category_roc', '-cr',
                        dest="category_roc",
                        action="store_true",
                        help='Use category roc mode to validate')

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

    if args.min_mode is False and args.real_labels is False and args.category_roc is False:
        raise RuntimeError("No validation mode has been selected, please choose at least one.")

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
        batch_count_n = batch_count_a = validation_config["exp_params"]["batch_count"]
    except KeyError:
        batch_count_n = len(normal_data_dataloader)
        batch_count_a = len(abnormal_data_dataloader)

    try:
        assert batch_count_n <= len(normal_data_dataloader) and batch_count_a <= len(abnormal_data_dataloader)
    except AssertionError:
        print("Chosen batch count '{}' or '{}' is larger than there are".format(batch_count_n, batch_count_a) +
              " available batches for the validation sets.")
        raise

    number_visualization_batches = validation_config["exp_params"]["number_visualization_batches"]

    """
    1. Durch Dataloader durchgehen und Batch in Padim eingeben -> 3 x Score Map bekommen
    2. Abspeichern ([3 x ScoreMap.max], label, prediction_label) (label ist das was vom Dataset kommt,
                                                prediction_label is 0 für normal, 1 für abnormal)
    3. Das für beide Dataloader machen, jeweils x visualization batches speichern -> also die score maps komplett
                                                                            abspeichern für die batches
    4. Für AUC_ROC 
        - Mit bekannten Labels nutzen, d.h. für jeden Eintrag nehme das ScoreMap.max raus bei dem label übereinstimm
        - Min -> Nimm die Minimum Score Map.max
        - So wie es jetzt ist -> Drei mal ROC berechnen dann gucken wo es rausfällt
        -> Mit resultierender Liste dann AUC_ROC
        
    5. Visualization batches:
        - Je nach gewähltem Verfahren in 4. nehme dann die jeweilige Score Map und nutze die zur Visualisierung
        - Wahlweise auch dreimal machen 
    
    
    """

    predictions_n, visualization_batches_n, visualization_score_maps_n = iterate_through_data(
        padim, normal_data_dataloader, batch_count_n, min_max_normalization, number_visualization_batches,
        abnormal_data=False
    )

    predictions_a, visualization_batches_a, visualization_score_maps_a = iterate_through_data(
        padim, abnormal_data_dataloader, batch_count_a, min_max_normalization, number_visualization_batches,
        abnormal_data=True
    )

    predictions_min_mode = []
    predictions_real_labels = []

    ground_truths = []

    for current_prediction in predictions_a + predictions_n:
        ground_truths.append(current_prediction[2])

        if args.min_mode:
            predictions_min_mode.append(current_prediction[0].min().cpu().numpy())

        if args.real_labels:
            index = current_prediction[1] % 3
            predictions_real_labels.append(current_prediction[0][index].cpu().numpy())

    image_savepath = os.path.join(args.experiment_dir, "validation")

    # Delete entries of the validation folder if it exists so that there is no overlap between validations
    try:
        shutil.rmtree(image_savepath)
    except FileNotFoundError:
        # Already deleted, pass
        pass

    os.makedirs(image_savepath, exist_ok=True)

    if args.min_mode:
        (fig, _), best_threshold = get_roc_plot_and_threshold(predictions_min_mode, ground_truths)
        fig.savefig(os.path.join(image_savepath, 'roc_curve_min_mode.png'), dpi=100)
        print("Saved ROC for min_mode to {}".format(image_savepath))

    if args.real_labels:
        (fig, _), best_threshold = get_roc_plot_and_threshold(predictions_real_labels, ground_truths)
        fig.savefig(os.path.join(image_savepath, 'roc_curve_real_labels.png'), dpi=100)
        print("Saved ROC for real_labels to {}".format(image_savepath))

    # v_max = float(max([x.max() for x in visualization_batches_n + visualization_batches_a]))
    # v_min = float(min([x.min() for x in visualization_batches_n + visualization_batches_a]))
    #
    # current_index_n = 0
    # current_index_a = len(predicted_category) // 2
    # for i in tqdm(range(len(visualization_batches_n))):
    #     scores_n = visualization_batches_n[i]
    #     scores_a = visualization_batches_a[i]
    #     new_scores_n = []
    #     new_scores_a = []
    #
    #     chosen_thresholds_n = []
    #     chosen_thresholds_a = []
    #
    #     predicted_category_n = predicted_category[current_index_n:current_index_n + batch_size]
    #     predicted_category_a = predicted_category[current_index_a:current_index_a + batch_size]
    #
    #     for j in range(scores_n.size(1)):
    #         new_scores_n.append(scores_n[predicted_category_n[j], j].cpu().numpy())
    #         new_scores_a.append(scores_a[predicted_category_a[j], j].cpu().numpy())
    #
    #         chosen_thresholds_n.append(category_thresholds[predicted_category_n[j]])
    #         chosen_thresholds_a.append(category_thresholds[predicted_category_a[j]])
    #
    #     gt_n = gt_n_tensor[i]
    #     batch_n = batch_normal[i]
    #     save_grid_plot(batch_n, i, gt_n, np.array(new_scores_n), chosen_thresholds_n, v_max, v_min, image_savepath,
    #                    backbone_kind=backbone_kind)
    #
    #     gt_a = gt_a_tensor[i]
    #     batch_a = batch_abnormal[i]
    #     save_grid_plot(batch_a, i, gt_a, np.array(new_scores_a), chosen_thresholds_a, v_max, v_min, image_savepath,
    #                    backbone_kind=backbone_kind)
    #
    #     current_index_n += batch_size
    #     current_index_a += batch_size
    #
    # print("Saved validation images to {}".format(image_savepath))


if __name__ == "__main__":
    main()
