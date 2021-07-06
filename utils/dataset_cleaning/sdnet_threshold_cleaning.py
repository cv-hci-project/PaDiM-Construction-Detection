import argparse
import multiprocessing as mp
import os
from shutil import copyfile, rmtree

import numpy as np
from PIL import Image


def get_mean(img_file_path):
    img = np.asarray(Image.open(img_file_path))
    return np.mean(img)


def clean_data_and_copy(data_root_dir, clean_data_root_dir, dirty_data_root_dir, category, abnormal_data: bool,
                        copy_dirty_data: bool = False):
    if not abnormal_data:
        folder_prefix = "U"
    else:
        folder_prefix = "C"

    clean_data_path = os.path.join(clean_data_root_dir, category, folder_prefix + category)
    os.makedirs(clean_data_path)

    data_path = os.path.join(data_root_dir, category, folder_prefix + category)
    data = np.array([os.path.join(data_path, img_file) for img_file in os.listdir(data_path)])

    means = []
    results = []

    with mp.Pool(8) as pool:

        for img_path in data:
            results.append(pool.apply_async(get_mean, args=(img_path,)))

        for result in results:
            means.append(result.get())

    # lower_threshold = np.mean(means) - np.std(means)
    # upper_threshold = np.mean(means) + np.std(means)

    lower_threshold = np.quantile(means, 0.05)
    upper_threshold = np.quantile(means, 0.95)

    clean_data = data[(means > lower_threshold) & (means < upper_threshold)]
    dirty_data = data[(means <= lower_threshold) | (means >= upper_threshold)]

    assert np.count_nonzero(clean_data) + np.count_nonzero(dirty_data) == data.shape[0]

    for clean_file in clean_data:
        copyfile(clean_file, os.path.join(clean_data_path, os.path.basename(clean_file)))

    if copy_dirty_data:
        dirty_data_path = os.path.join(dirty_data_root_dir, category, folder_prefix + category)
        os.makedirs(dirty_data_path)

        for dirty_file in dirty_data:
            copyfile(dirty_file, os.path.join(dirty_data_path, os.path.basename(dirty_file)))

    return clean_data_path


def main():
    parser = argparse.ArgumentParser(description='Clean the SDNET2018 dataset using thresholding')
    parser.add_argument('--dataset', '-d',
                        dest="root_dir",
                        metavar='DIR',
                        help='Path to the SDNET2018 dataset')

    parser.add_argument('--clean_dataset_path', '-c',
                        dest="clean_root_dir",
                        metavar='DIR',
                        help='Path where the cleaned data shall be saved, attention files in there will be deleted')

    args = parser.parse_args()

    dirty_root_dir = "/tmp/temp-dirty-percentile/"

    try:
        rmtree(args.clean_root_dir)
        rmtree(dirty_root_dir)
    except FileNotFoundError:
        # Already deleted, pass
        pass

    os.makedirs(args.clean_root_dir)
    os.makedirs(dirty_root_dir)

    classes = ["D", "P", "W"]

    for _class in classes:
        clean_data_path = clean_data_and_copy(data_root_dir=args.root_dir, clean_data_root_dir=args.clean_root_dir,
                                              dirty_data_root_dir=dirty_root_dir, category=_class, abnormal_data=False,
                                              copy_dirty_data=False)

        print("Copied clean data for category '{}' to {}".format(_class, clean_data_path))

    # Copy images with cracks without changing them
    for _class in classes:
        clean_data_path = clean_data_and_copy(data_root_dir=args.root_dir, clean_data_root_dir=args.clean_root_dir,
                                              dirty_data_root_dir=dirty_root_dir, category=_class, abnormal_data=True,
                                              copy_dirty_data=False)

        print("Copied anomaly data for category '{}' to {}".format(_class, clean_data_path))

    print("Done")


if __name__ == "__main__":
    main()
