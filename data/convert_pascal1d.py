import os
import numpy as np
import pickle
import cv2
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, default="./Pascal1D", help="root directory of the data"
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="train faction and the rest is validation faction",
    )
    args, _ = parser.parse_known_args()

    return args


def convert_pascal1d(x, y, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_pascal1d"
    info_json["total_categories"] = x.shape[0]
    info_json["minimum_images_per_category"] = x.shape[1]
    info_json["median_images_per_category"] = x.shape[1]
    info_json["maximum_images_per_category"] = x.shape[1]

    with open(os.path.join(output_path, "label.csv"), "w") as f:
        f.write("FILE_NAME,CATEGORY,REGRESSION_LABEL\n")
        for category in range(x.shape[0]):
            for obj in range(x.shape[1]):
                image_name = str(category) + "_" + str(obj) + ".jpg"
                image_path = os.path.join(output_path, "images/" + image_name)
                cv2.imwrite(image_path, x[category][obj])
                f.write(
                    image_name
                    + ","
                    + str(category)
                    + ","
                    + str(float(y[category][obj]))
                    + "\n"
                )
                info_json["file_name"].append(image_name)
                info_json["category"].append(category)
                info_json["regression_label"].append(float(y[category][obj]))

    b = json.dumps(info_json)
    info_f.write(b)
    info_f.close()


if __name__ == "__main__":
    args = parse_arguments()

    pickle_name = "train_data.pkl"

    x, y = pickle.load(open(os.path.join(args.root_dir, pickle_name), "rb"))
    x, y = np.array(x), np.array(y)
    y = y[:, :, -1, None]

    split_index = int(x.shape[0] * args.train_fraction)
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_val = x[split_index:]
    y_val = y[split_index:]

    # Pascal1D only has train and val split, we split val split from train
    convert_pascal1d(x=x_train, y=y_train, output_path="./Pascal1D_train")
    convert_pascal1d(x=x_val, y=y_val, output_path="./Pascal1D_val")

    pickle_name = "val_data.pkl"

    x, y = pickle.load(open(os.path.join(args.root_dir, pickle_name), "rb"))
    x, y = np.array(x), np.array(y)
    y = y[:, :, -1, None]

    convert_pascal1d(x=x, y=y, output_path="./Pascal1D_test")
