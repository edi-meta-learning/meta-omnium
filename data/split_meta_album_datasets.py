# -----------------------
# Imports
# -----------------------
import argparse
import json
import os
import shutil

import numpy as np


# -----------------------
# Arguments
# -----------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", type=str, required=True, help="which dataset to use"
    )
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of the data"
    )

    args, _ = parser.parse_known_args()

    return args


def train_test_split_classes(classes, test_split=0.3):
    classes = np.unique(np.array(classes))
    np.random.shuffle(classes)
    cut_off = int((1 - test_split) * len(classes))
    return classes[:cut_off], classes[cut_off:]


if __name__ == "__main__":
    args = parse_arguments()

    with open(os.path.join(args.root_dir, args.dataset, "info.json"), "r") as f:
        info = json.load(f)

    classes = info["category"]

    np.random.seed(0)
    if "Micro" in args.dataset:
        train_classes, eval_classes = train_test_split_classes(classes, test_split=0.5)
    else:
        train_classes, eval_classes = train_test_split_classes(classes, test_split=0.3)
    val_classes, test_classes = train_test_split_classes(eval_classes, test_split=0.5)

    categories_dict = {
        "Trn": set(train_classes),
        "Val": set(val_classes),
        "Test": set(test_classes),
    }

    for split in ["Trn", "Val", "Test"]:
        file_name_list = []
        category_list = []
        segmentation_file_name_list = []
        os.mkdir(os.path.join(args.root_dir, args.dataset + "_" + split))
        os.mkdir(os.path.join(args.root_dir, args.dataset + "_" + split, "images"))

        for idx, file_name in enumerate(info["file_name"]):
            if info["category"][idx] in categories_dict[split]:
                source = os.path.join(args.root_dir, args.dataset, "images", file_name)
                target = os.path.join(
                    args.root_dir, args.dataset + "_" + split, "images", file_name
                )
                shutil.copy(source, target)
                file_name_list.append(file_name)
                category_list.append(info["category"][idx])

        # Create the dictionary with relevant information
        num_classes = len(categories_dict[split])
        info_path = os.path.join(args.root_dir, args.dataset + "_" + split, "info.json")
        updated_info = {
            "dataset_name": args.dataset + "_" + str(split),
            "dataset_description": args.dataset
            + "_"
            + str(split)
            + " - Few-Shot Classification Dataset",
            "task_type": "classification",
            "total_categories": num_classes,
            "total_super_categories": 0,
            "uniform_number_of_images_per_category": info[
                "uniform_number_of_images_per_category"
            ],
            "minimum_images_per_category": info["minimum_images_per_category"],
            "median_images_per_category": info["median_images_per_category"],
            "maximum_images_per_category": info["maximum_images_per_category"],
            "has_super_categories": info["has_super_categories"],
            "file_name": file_name_list,
            "category": category_list,
        }

        # Store the information as a json file
        with open(info_path, "w") as f:
            json.dump(updated_info, f)
