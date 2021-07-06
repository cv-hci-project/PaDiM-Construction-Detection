import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import morphology
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def get_roc_plot_and_threshold(predictions, gt_list):
    # calculate image-level ROC AUC score
    # img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
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
