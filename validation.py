import pickle
from torch import Tensor
from tqdm import tqdm
import matplotlib
import numpy as np
import torch
import yaml
import os
import argparse
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage import morphology

from backbones import backbone_models
from utils.dataloader_utils import get_dataloader
from utils.utils import transforms_for_pretrained, get_embedding, calculate_score_map, get_roc_plot_and_threshold

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = x.numpy()
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x

def create_mask(img_score: np.ndarray, threshold):
    idx_above_threshold = img_score > threshold
    idx_below_threshold = img_score <= threshold

    mask = img_score
    mask[idx_above_threshold] = 1
    mask[idx_below_threshold] = 0

    kernel = morphology.disk(4)
    mask = morphology.opening(mask, kernel)
    # mask *= 255

    return mask

def create_img_subplot(img, img_score, threshold, vmin, vmax):
    img = denormalization(img)
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
        name = "Validation_{}_Image_Classified_as_{}_{}.png".format(int(label[i]), classified_as, i+batch_id*num)
        fig_img.savefig(os.path.join(path, name), dpi=100)

def validation_step(batch, backbone, number_of_embeddings, number_of_patches, _means, _covs, crop_size, device):

    batch = batch.to(device)
    with torch.no_grad():
        features_1, features_2, features_3 = backbone(batch)

    embedding_ids = torch.randperm(backbone.embeddings_size)[:number_of_embeddings].to(device)
    embeddings = get_embedding(features_1, features_2, features_3, embedding_ids, device)
    B, C, H, W = embeddings.size()
    embeddings = embeddings.view(-1, number_of_embeddings, number_of_patches)

    scores = calculate_score_map(embeddings.cpu(), (B, C, H, W), _means.cpu(), _covs.cpu(),
                                 crop_size, min_max_norm=True)
    return torch.Tensor(scores)
def main():
    parser = argparse.ArgumentParser(description='Validate a PaDiM model')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='Path to the configuration file',
                        default='configurations/validation.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    trained_feature_filepath = os.path.join(config['logging_params']['save_dir'], 'train_%s.pkl' % config['logging_params']['name'])
    image_savepath = config['logging_params']['img_save_dir']
    number_of_embeddings = config['exp_params']["number_of_embeddings"]
    crop_size = config["exp_params"]["crop_size"]
    batch_number = config["exp_params"]["batch_number"]
    batch_size = config["exp_params"]["batch_size"]


    # load model
    print('load train set feature from: %s' % trained_feature_filepath)
    with open(trained_feature_filepath, 'rb') as f:
        _means, _covs, = pickle.load(f)

    # choose device
    gpu_id = config["trainer_params"]["gpu"]
    if gpu_id >= 0:
        device = torch.device('cuda:{}'.format(gpu_id))

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Chose gpu_id '{}', but no GPU is available. If you want to use the CPU, set it to '-1'".format(gpu_id))
    else:
        device = torch.device('cpu')

    print("Device in use: {}".format(device))

    # load backbone model
    backbone = backbone_models[config['exp_params']['backbone']]()
    backbone.to(device)
    number_of_patches = backbone.num_patches

    # load test data
    transform = transforms_for_pretrained(crop_size=crop_size)
    normal_data_dataloader = iter(get_dataloader(config['exp_params'], train_split=False, abnormal_data=False,
                                            transform=transform))
    abnormal_data_dataloader = iter(get_dataloader(config['exp_params'], train_split=False, abnormal_data=True,
                                            transform=transform))

    gt_n_tensor = torch.zeros((batch_number, batch_size, 1))
    gt_a_tensor = torch.ones((batch_number, batch_size, 1))
    scores_n_tensor = torch.zeros((batch_number, batch_size, crop_size, crop_size))
    scores_a_tensor = torch.zeros((batch_number, batch_size, crop_size, crop_size))
    batch_normal = torch.zeros((batch_number, batch_size, 3, crop_size, crop_size))
    batch_abnormal = torch.zeros((batch_number, batch_size, 3, crop_size, crop_size))
    # calculate score map
    for i in tqdm(range(batch_number)):
        batch_n = next(normal_data_dataloader)[0]
        batch_a = next(abnormal_data_dataloader)[0]
        batch_normal[i] = batch_n
        batch_abnormal[i] = batch_a
        scores_n_tensor[i] = validation_step(batch_n, backbone, number_of_embeddings, number_of_patches, _means, _covs, crop_size, device)
        scores_a_tensor[i] = validation_step(batch_a, backbone, number_of_embeddings, number_of_patches, _means, _covs, crop_size, device)

    scores_all = torch.cat([scores_n_tensor, scores_a_tensor], 0)
    bn, bz, cs1, cs2 = scores_all.shape
    scores_all = scores_all.reshape(bn*bz, cs1, cs2)

    gt_all = torch.cat([gt_n_tensor, gt_a_tensor], 0)
    gt_all = gt_all.reshape(bn*bz, 1)

    v_max = scores_all.max()
    v_min = scores_all.min()

    # calculate metrics
    (fig, _), best_threshold = get_roc_plot_and_threshold(scores_all, gt_all)
    fig.savefig(os.path.join(image_savepath, 'roc_curve.png'), dpi=100)
    print("Saved ROC  images to {}".format(image_savepath))



    for i in tqdm(range(batch_number)):
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
