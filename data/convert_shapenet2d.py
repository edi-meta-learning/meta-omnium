import os
import numpy as np
import pickle
import cv2
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_path",
        type=str,
        default="../data/ShapeNet3D_azi180ele30",
        help="root directory of the data",
    )
    parser.add_argument("--seed", type=int, default=2578, help="random seed")
    parser.add_argument("--num_instances_per_item", type=int, default=36, help="")

    args, _ = parser.parse_known_args()

    return args


def convert_shapenet2d(split, root_path, output_path):
    data = pickle.load(
        open(os.path.join(root_path, "shapenet3d_azi180ele30_" + split + ".pkl"), "rb")
    )
    images = data["images"]
    item_indices = data["item_indices"]
    Q = data["Q"]

    categories = np.unique(item_indices)
    num_instances_per_category = int(images.shape[0] / categories.shape[0])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_shapenet2d"
    info_json["total_categories"] = categories.shape[0]
    info_json["minimum_images_per_category"] = num_instances_per_category
    info_json["median_images_per_category"] = num_instances_per_category
    info_json["maximum_images_per_category"] = num_instances_per_category

    with open(os.path.join(output_path, "label.csv"), "w") as f:
        f.write("FILE_NAME,CATEGORY,REGRESSION_LABEL\n")
        for category in range(categories.shape[0]):
            for obj in range(num_instances_per_category):
                image_name = str(category) + "_" + str(obj) + ".jpg"
                image_path = os.path.join(output_path, "images/" + image_name)
                cv2.imwrite(image_path, images[category * obj, :, :, :3])
                regression_label = Q[category * obj].tolist()
                regression_label = [str(i) for i in regression_label]
                regression_label = "_".join(regression_label)
                f.write(
                    image_name + "," + str(category) + "," + regression_label + "\n"
                )
                info_json["file_name"].append(image_name)
                info_json["category"].append(category)
                info_json["regression_label"].append(regression_label)

    b = json.dumps(info_json)
    info_f.write(b)
    info_f.close()


if __name__ == "__main__":
    args = parse_arguments()

    convert_shapenet2d(
        split="test", root_path=args.root_path, output_path="./ShapeNet2D_test"
    )
