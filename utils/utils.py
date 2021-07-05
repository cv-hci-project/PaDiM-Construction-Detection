import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from skimage import morphology
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve


def get_roc_plot_and_threshold(predictions, gt_list):
    predicted_category = None
    category_thresholds = None

    if isinstance(predictions, dict):
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

        predictions = calculated_predictions

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

    return (fig, ax), best_threshold, predicted_category, category_thresholds


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
