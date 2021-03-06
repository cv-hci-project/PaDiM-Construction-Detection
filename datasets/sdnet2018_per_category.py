import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class SDNet2018PerCategory(Dataset):
    """
    SDNet2018 dataset which contains images of cracked and non-cracked concrete bridge decks, walls and pavements.

    Maguire, M., Dorafshan, S., & Thomas, R. J. (2018). SDNET2018: A concrete crack image dataset for machine learning
     applications. Utah State University. https://doi.org/10.15142/T3TD19
    """

    def __init__(self, root_dir, split: str = "train", abnormal_data: bool = False, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images, it assumes 'D', 'P' and 'W' as subfolders and these in
                turn contain subfolders with cracked and non-cracked images
            split (string): 'train', 'val' or 'test'
            abnormal_data (bool, optional): If this is true, the abnormal data is returned, i.e. images with cracks.
                Otherwise non-cracked images are returned.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        self.possible_splits = [
            "train",
            "val",
            "test"
        ]

        assert split in self.possible_splits, "Chosen split '{}' is not valid".format(split)

        self.split = split
        self.abnormal_data = abnormal_data

        self.data_directories_normal = [
            [os.path.join(self.root_dir, "D", "UD", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "D", "UD"))],
            [os.path.join(self.root_dir, "P", "UP", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "P", "UP"))],
            [os.path.join(self.root_dir, "W", "UW", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "W", "UW"))]
        ]

        self.data_directories_abnormal = [
            [os.path.join(self.root_dir, "D", "CD", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "D", "CD"))],
            [os.path.join(self.root_dir, "P", "CP", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "P", "CP"))],
            [os.path.join(self.root_dir, "W", "CW", _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, "W", "CW"))]
        ]

        normal_data, normal_category_indices = self.create_splits(self.data_directories_normal, abnormal_data=False)

        self.train_data_normal, self.val_data_normal, self.test_data_normal = normal_data
        self.train_category_indices_normal, self.val_category_indices_normal, self.test_category_indices_normal = normal_category_indices

        abnormal_data, abnormal_category_indices = self.create_splits(self.data_directories_abnormal, abnormal_data=True)

        self.train_data_abnormal, self.val_data_abnormal, self.test_data_abnormal = abnormal_data
        self.train_category_indices_abnormal, self.val_category_indices_abnormal, self.test_category_indices_abnormal = abnormal_category_indices

        self.train_data_normal_length = len(self.train_data_normal)
        self.val_data_normal_length = len(self.val_data_normal)
        self.test_data_normal_length = len(self.test_data_normal)

        self.train_data_abnormal_length = len(self.train_data_abnormal)
        self.val_data_abnormal_length = len(self.val_data_abnormal)
        self.test_data_abnormal_length = len(self.test_data_abnormal)

    def create_splits(self, data_directories: list, abnormal_data: bool):
        """
        Create train, val and test splits from the provided data_directory. This expects that data_directories is a list
        of subfolders and each subfolder contains direct paths to the image files. This way the train, val and test
        splits can be done per subfolder.

        Doing splits on all subfolders concatenated would result in the test set containing always the last portion of
        the concatenated list. This is not desired as then it would contain images of only one 'class' (i.e. bridge
        decks, walls or pavements). Also the other splits would potentially not see data from this 'class'.
        """
        train_split = []
        val_split = []
        test_split = []

        train_indices_category = {}
        val_indices_category = {}
        test_indices_category = {}

        train_old_index = 0
        val_old_index = 0
        test_old_index = 0

        for i, sub_dir in enumerate(data_directories):
            train_index = round(len(sub_dir) * self.train_split)
            val_index = round(len(sub_dir) * self.val_split)

            train_split += sub_dir[:train_index]
            val_split += sub_dir[train_index:train_index + val_index]
            test_split += sub_dir[train_index + val_index:]

            if not abnormal_data:
                category = i
            else:
                category = i + 3

            train_indices_category[category] = train_old_index + len(sub_dir[:train_index])
            val_indices_category[category] = val_old_index + len(sub_dir[train_index:train_index + val_index])
            test_indices_category[category] = test_old_index + len(sub_dir[train_index + val_index:])

            train_old_index += len(sub_dir[:train_index])
            val_old_index += len(sub_dir[train_index:train_index + val_index])
            test_old_index += len(sub_dir[train_index + val_index:])

        return (train_split, val_split, test_split), (train_indices_category, val_indices_category, test_indices_category)

    def __len__(self):
        if not self.abnormal_data:
            if self.split == "train":
                return self.train_data_normal_length
            elif self.split == "val":
                return self.val_data_normal_length
            else:
                return self.test_data_normal_length
        else:
            if self.split == "train":
                return self.train_data_abnormal_length
            elif self.split == "val":
                return self.val_data_abnormal_length
            else:
                return self.test_data_abnormal_length

    def get_label(self, idx, category_indices: dict):
        for k, v in category_indices.items():
            if idx < v:
                return k

        raise RuntimeError("The chosen index '{}' was not registered in the category indice dict.")

    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, list):
            raise RuntimeError(
                "SDNet2018 Dataset was accessed with a list of indices, not sure if this works. Aborting")

        if not self.abnormal_data:
            if self.split == "train":
                data = self.train_data_normal
                label = self.get_label(idx, self.train_category_indices_normal)
            elif self.split == "val":
                data = self.val_data_normal
                label = self.get_label(idx, self.val_category_indices_normal)
            else:
                data = self.test_data_normal
                label = self.get_label(idx, self.test_category_indices_normal)
        else:
            if self.split == "train":
                data = self.train_data_abnormal
                label = self.get_label(idx, self.train_category_indices_abnormal)
            elif self.split == "val":
                data = self.val_data_abnormal
                label = self.get_label(idx, self.val_category_indices_abnormal)
            else:
                data = self.test_data_abnormal
                label = self.get_label(idx, self.test_category_indices_abnormal)

        img = Image.open(data[idx])

        if self.transform:
            img = self.transform(img)

        return img, label


class SDNet2018OnlyOneCategory(Dataset):
    def __init__(self, root_dir, category: str, split: str = "train", abnormal_data: bool = False, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.category = category

        self.train_split = 0.65
        self.val_split = 0.15
        self.test_split = 0.2

        self.possible_splits = [
            "train",
            "val",
            "test"
        ]

        assert split in self.possible_splits, "Chosen split '{}' is not valid".format(split)

        self.split = split
        self.abnormal_data = abnormal_data

        self.data_directories_normal = [
            os.path.join(self.root_dir, self.category, "U" + self.category, _dir) for _dir in
             os.listdir(os.path.join(self.root_dir, self.category, "U" + self.category))
        ]

        self.data_directories_abnormal = [
            os.path.join(self.root_dir, self.category, "C" + self.category, _dir) for _dir in
            os.listdir(os.path.join(self.root_dir, self.category, "C" + self.category))
        ]

        self.train_index_normal = round(len(self.data_directories_normal) * self.train_split)
        self.val_index_normal = round(len(self.data_directories_normal) * self.val_split)

        self.train_data_normal = self.data_directories_normal[:self.train_index_normal]
        self.val_data_normal = self.data_directories_normal[self.train_index_normal:self.train_index_normal + self.val_index_normal]
        self.test_data_normal = self.data_directories_normal[self.train_index_normal + self.val_index_normal:]

        self.train_index_abnormal = round(len(self.data_directories_abnormal) * self.train_split)
        self.val_index_abnormal = round(len(self.data_directories_abnormal) * self.val_split)

        self.train_data_abnormal = self.data_directories_abnormal[:self.train_index_abnormal]
        self.val_data_abnormal = self.data_directories_abnormal[self.train_index_abnormal:self.train_index_abnormal + self.val_index_abnormal]
        self.test_data_abnormal = self.data_directories_abnormal[self.train_index_abnormal + self.val_index_abnormal:]

    def __len__(self):
        if self.split == "train":
            if not self.abnormal_data:
                return len(self.train_data_normal)
            else:
                return len(self.train_data_abnormal)
        elif self.split == "val":
            if not self.abnormal_data:
                return len(self.val_data_normal)
            else:
                return len(self.val_data_abnormal)
        else:
            if not self.abnormal_data:
                return len(self.test_data_normal)
            else:
                return len(self.test_data_abnormal)

    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, list):
            raise RuntimeError(
                "SDNet2018 Dataset was accessed with a list of indices, not sure if this works. Aborting")

        if not self.abnormal_data:
            label = 0
            if self.split == "train":
                data = self.train_data_normal
            elif self.split == "val":
                data = self.val_data_normal
            else:
                data = self.test_data_normal
        else:
            label = 1
            if self.split == "train":
                data = self.train_data_abnormal
            elif self.split == "val":
                data = self.val_data_abnormal
            else:
                data = self.test_data_abnormal

        img = Image.open(data[idx])

        if self.transform:
            img = self.transform(img)

        return img, label