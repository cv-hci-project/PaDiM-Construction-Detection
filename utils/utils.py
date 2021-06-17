import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


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


def get_embedding(features_1, features_2, features_3, embedding_ids):
    embedding = features_1
    embedding = _embedding_concat(embedding, features_2)
    embedding = _embedding_concat(embedding, features_3)

    # Select a random amount of embeddings
    embedding = torch.index_select(embedding, dim=1, index=embedding_ids)

    return embedding
