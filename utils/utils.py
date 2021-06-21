import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def get_roc_plot_and_threshold(scores, gt_list):
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)

    fpr, tpr, thresholds = roc_curve(gt_list, img_scores[0])
    img_roc_auc = roc_auc_score(gt_list, img_scores[0])

    fig, ax = plt.subplots(1, 1)
    fig_img_rocauc = ax

    fig_img_rocauc.plot(fpr, tpr, label="ROC Curve (area = {:.2f})".format(img_roc_auc))
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")

    precision, recall, thresholds = precision_recall_curve(gt_list, img_scores[0])
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_threshold = thresholds[np.argmax(f1)]

    return (fig, ax), best_threshold


def denormalization_for_pretrained(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def transforms_for_pretrained(crop_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([transforms.CenterCrop(crop_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean,
                                                    std=std)])


def _embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def get_embedding(features_1, features_2, features_3, embedding_ids, device):
    embedding = features_1
    embedding = _embedding_concat(embedding, features_2).to(device)
    embedding = _embedding_concat(embedding, features_3).to(device)

    # Select a random amount of embeddings
    embedding = torch.index_select(embedding, dim=1, index=embedding_ids)

    return embedding

def _calculate_dist_list(embedding, embedding_dimensions: tuple, means, covs):
    B, C, H, W = embedding_dimensions

    dist_list = []
    for i in range(H * W):
        mean = means[i, :]
        conv_inv = np.linalg.inv(covs[i, :, :])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose((1, 0)).reshape((B, H, W))

    return dist_list

def calculate_score_map(embedding, embedding_dimensions: tuple, means, covs, crop_size, min_max_norm: bool):
    dist_list = _calculate_dist_list(embedding, embedding_dimensions, means, covs)

    # Upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=crop_size, mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # Apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    if min_max_norm:
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
    else:
        scores = score_map

    return scores
