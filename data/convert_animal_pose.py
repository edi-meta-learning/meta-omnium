import os
import numpy as np
import cv2
import json
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, default="./", help="root directory of the data"
    )
    args, _ = parser.parse_known_args()
    return args


def convert_animal_pose(split, classes, root_path, output_path):
    data = json.load(open(os.path.join(root_path, "keypoints.json")))

    # Classes are folder names in png4 folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_pose_animals"
    info_json["total_categories"] = len(classes)

    info_json["minimum_images_per_category"] = 72
    info_json["median_images_per_category"] = 72
    info_json["maximum_images_per_category"] = 72
    info_json["num_keypoints"] = 20

    iteration = 1
    for i in range(0, len(data["annotations"])):
        image_id = str(data["annotations"][i]["image_id"])
        image_name = data["images"][image_id]
        image_new_name = str(iteration) + ".jpg"
        iteration += 1
        category = int(data["annotations"][i]["category_id"]) - 1
        if category in classes:
            source = os.path.join(root_path, "images", image_name)
            target = os.path.join(output_path, "images", image_new_name)
            source_image = cv2.imread(source)
            bbox = data["annotations"][i]["bbox"]
            cropped_image = source_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            keypoint = np.array(data["annotations"][i]["keypoints"]).tolist()
            new_keypoint_list = []
            # print("orig keypoints", keypoint, bbox)
            for i in range(len(keypoint)):
                if keypoint[i][0] != 0 and keypoint[i][1] != 0:
                    new_keypoint = [keypoint[i][0] - bbox[0], keypoint[i][1] - bbox[1]]
                else:
                    new_keypoint = [keypoint[i][0], keypoint[i][1]]
                new_keypoint_list.append(new_keypoint)

            cv2.imwrite(target, cropped_image)
            info_json["category"].append(category)
            info_json["file_name"].append(image_new_name)
            info_json["regression_label"].append(new_keypoint_list)

    print(info_json)
    b = json.dumps(info_json)
    info_f.write(b)
    info_f.close()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    num_classes = 5
    train_classes = [0, 1]
    val_classes = [2, 3]
    test_classes = [4]

    print("Making train data", train_classes)
    convert_animal_pose(
        split="train",
        classes=train_classes,
        root_path=args.root_dir,
        output_path="./Animal_pose1" + "_train",
    )

    print("Making val data", val_classes)
    convert_animal_pose(
        split="val",
        classes=val_classes,
        root_path=args.root_dir,
        output_path="./Animal_pose1_val",
    )

    print("Making test data", test_classes)
    convert_animal_pose(
        split="test",
        classes=test_classes,
        root_path=args.root_dir,
        output_path="./Animal_pose1_test",
    )
