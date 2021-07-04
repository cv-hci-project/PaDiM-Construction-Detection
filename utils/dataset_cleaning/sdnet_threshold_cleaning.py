import argparse
import multiprocessing as mp
import os
from shutil import copyfile, rmtree

import numpy as np
from PIL import Image


def get_mean(img_file_path):
    img = np.asarray(Image.open(img_file_path))
    return np.mean(img)


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
        clean_data_path = os.path.join(args.clean_root_dir, _class, "U" + _class)
        os.makedirs(clean_data_path)

        data_path = os.path.join(args.root_dir, _class, "U" + _class)
        data = np.array([os.path.join(data_path, img_file) for img_file in os.listdir(data_path)])

        means = []
        results = []

        with mp.Pool(8) as pool:

            for img_path in data:
                results.append(pool.apply_async(get_mean, args=(img_path, )))

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

        dirty_data_path = os.path.join(dirty_root_dir, _class, "U" + _class)
        os.makedirs(dirty_data_path)

        for dirty_file in dirty_data:
            copyfile(dirty_file, os.path.join(dirty_data_path, os.path.basename(dirty_file)))

        print("Copied clean data for category '{}' to {}".format(_class, clean_data_path))

    # Copy images with cracks without changing them
    for _class in classes:
        new_data_path = os.path.join(args.clean_root_dir, _class, "C" + _class)
        os.makedirs(new_data_path)

        data_path_anomalies = os.path.join(args.root_dir, _class, "C" + _class)
        data_anomalies = [os.path.join(data_path_anomalies, img_file) for img_file in os.listdir(data_path_anomalies)]

        for anomaly_img in data_anomalies:
            copyfile(anomaly_img, os.path.join(new_data_path, os.path.basename(anomaly_img)))

        print("Copied anomaly data without changing them for category '{}' to {}".format(_class, new_data_path))

    print("Done")


if __name__ == "__main__":
    main()
