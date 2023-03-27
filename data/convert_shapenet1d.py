import os
import numpy as np
import pickle
import cv2
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, default="./", help="root directory of the data"
    )
    parser.add_argument(
        "--data_size",
        type=str,
        default="large",
        help="data size for train (large|middle|small)",
    )
    args, _ = parser.parse_known_args()

    return args


def convert_shapenet1d(split, root_path, output_path):
    if split == "train":
        pickle_name = f"{split}_data_{args.data_size}.pkl"
    else:
        pickle_name = f"{split}_data.pkl"

    x, y = pickle.load(open(os.path.join(root_path, pickle_name), "rb"))
    x, y = np.array(x), np.array(y)
    y = y[:, :, -1, None]

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_shapenet1d"
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

    convert_shapenet1d(
        split="train",
        root_path=args.root_dir,
        output_path="./ShapeNet1D_" + args.data_size + "_train",
    )
    convert_shapenet1d(
        split="val", root_path=args.root_dir, output_path="./ShapeNet1D_val"
    )
    convert_shapenet1d(
        split="test", root_path=args.root_dir, output_path="./ShapeNet1D_test"
    )