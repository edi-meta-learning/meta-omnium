import os
import numpy as np
import cv2
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_path",
        type=str,
        default="../../what-matters-for-meta-learning/data/distractor",
        help="root directory of the data",
    )
    parser.add_argument(
        "--data_size",
        type=str,
        default="large",
        help="data size for train (large|middle|small)",
    )
    parser.add_argument("--seed", type=int, default=2578, help="random seed")
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="train faction and the rest is validation faction",
    )
    parser.add_argument("--image_height", type=int, default=128, help="")
    parser.add_argument("--image_width", type=int, default=128, help="")
    parser.add_argument("--image_channel", type=int, default=1, help="")
    parser.add_argument("--num_instances_per_item", type=int, default=36, help="")

    args, _ = parser.parse_known_args()

    return args


def extract_data(data, num_instances_per_item, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    label_f = open(os.path.join(output_path, "label.csv"), "w")
    label_f.write("FILE_NAME,CATEGORY,Center,Angle\n")

    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["center"] = list()
    info_json["angle"] = list()
    info_json["task_type"] = "regression_distractor"
    info_json["total_categories"] = 12
    info_json["minimum_images_per_category"] = num_instances_per_item
    info_json["median_images_per_category"] = num_instances_per_item
    info_json["maximum_images_per_category"] = num_instances_per_item

    images, item_indices, item_angles, centers = [], [], [], []
    for item_index, item in enumerate(data):
        for m, instance in enumerate(item):
            images.append(instance[0])
            item_indices.append(item_index)
            centers.append(instance[3])

            # Save image
            image = np.reshape(
                np.array(instance[0]),
                (args.image_height, args.image_width, args.image_channel),
            )
            image = 255 - (image * 255).astype(np.uint8)
            image_name = str(item_index) + "_" + str(m) + ".jpg"
            image_path = os.path.join(output_path, "images/" + image_name)
            cv2.imwrite(image_path, image)
            info_json["file_name"].append(image_name)

            # Save center
            info_json["center"].append(instance[3])

            # Save angle
            degrees_per_increment = 360.0 / num_instances_per_item
            angle = instance[2] * degrees_per_increment
            angle_radians = np.deg2rad(angle)
            angle_label = [angle, np.sin(angle_radians), np.cos(angle_radians)]
            info_json["angle"].append(angle_label)

            info_json["category"].append(item_index)

            label_f.write(
                image_name
                + ","
                + str(item_index)
                + ","
                + "_".join([str(i) for i in info_json["center"][-1]])
                + ","
                + "_".join([str(i) for i in info_json["angle"][-1]])
                + ","
                + "\n"
            )

    b = json.dumps(info_json)
    info_f.write(b)
    info_f.close()


if __name__ == "__main__":
    args = parse_arguments()
    train_categories = [
        "02691156",
        "02828884",
        "02933112",
        "02958343",
        "02992529",
        "03001627",
        "03211117",
        "03636649",
        "03691459",
        "04379243",
    ]
    test_categories = ["04256520", "04530566"]

    for category in train_categories:
        file = os.path.join(args.root_path, "{0:s}_multi.npy".format(category))
        if category == train_categories[0]:  # first time through
            data_train = np.load(file, allow_pickle=True)

        else:
            data_train_new = np.load(file, allow_pickle=True)
            data_train = np.concatenate((data_train, data_train_new), axis=0)

    for category in test_categories:
        file = os.path.join(args.root_path, "{0:s}_multi.npy".format(category))
        if category == test_categories[0]:  # first time through
            data_test = np.load(file, allow_pickle=True)
        else:
            data_test_new = np.load(file, allow_pickle=True)
            data_test = np.concatenate((data_test, data_test_new), axis=0)

    total_train_items = data_train.shape[0]
    num_instances_per_item = data_train.shape[1]

    # rest is val dataset
    train_size = (int)(args.train_fraction * total_train_items)

    test_size = data_test.shape[0]
    np.random.seed(args.seed)
    np.random.shuffle(data_train)

    extract_data(
        data=data_train[:train_size],
        num_instances_per_item=args.num_instances_per_item,
        output_path="./distractor_train",
    )

    extract_data(
        data=data_train[train_size:],
        num_instances_per_item=args.num_instances_per_item,
        output_path="./distractor_val",
    )

    extract_data(
        data=data_test,
        num_instances_per_item=args.num_instances_per_item,
        output_path="./distractor_test",
    )
