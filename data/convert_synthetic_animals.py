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


# bbox using keypoints
def cal_bbox(pts):
    bbox_list = []
    y_min = int(np.min(pts[:, 1]))
    y_max = int(np.max(pts[:, 1]))
    x_min = int(np.min(pts[:, 0]))
    x_max = int(np.max(pts[:, 0]))
    bbox_list = [x_min, x_max, y_min, y_max]
    return bbox_list


def convert_synthetic_pose(split, classes, root_path, output_path):
    class_to_idx = {"0": "tiger", "1": "horse"}

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_pose_animals_syn"
    info_json["total_categories"] = len(classes)
    info_json["minimum_images_per_category"] = 72
    info_json["median_images_per_category"] = 72
    info_json["maximum_images_per_category"] = 72
    info_json["num_keypoints"] = 18

    for class_id in classes:
        folder = class_to_idx[str(class_id)] + "_combineds5r5_texture"
        images_list = [
            file
            for file in os.listdir(os.path.join(root_path, folder))
            if file.endswith("img.png")
        ]

        for image_name in images_list:
            keypt_file = image_name.split(".png")[0] + ".png_kpts.npy"
            pts = np.load(os.path.join(root_path, folder, keypt_file))
            animal = class_to_idx[str(class_id)]
            if animal == "horse":
                idxs = np.array(
                    [
                        1718,
                        1684,
                        1271,
                        1634,
                        1650,
                        1643,
                        1659,
                        925,
                        392,
                        564,
                        993,
                        726,
                        1585,
                        1556,
                        427,
                        1548,
                        967,
                        877,
                    ]
                )  # selected kpts w.r.t. the TigDog annotations
                idxs_mask = np.zeros(3299)  # for horse
            elif animal == "tiger":
                idxs = np.array(
                    [
                        2753,
                        2679,
                        2032,
                        1451,
                        1287,
                        3085,
                        1632,
                        229,
                        1441,
                        1280,
                        2201,
                        1662,
                        266,
                        158,
                        270,
                        152,
                        219,
                        129,
                    ]
                )
                idxs_mask = np.zeros(3299)

            pts *= pts[:, 2].reshape(-1, 1)
            pts = pts[idxs][:, :2]
            indices = np.where(np.all([pts[:, 0] > 0, pts[:, 1] > 0], axis=0))
            visible_joints = pts[indices].reshape(-1, 2)
            image = cv2.imread(
                os.path.join(os.path.join(root_path, folder, image_name))
            )
            bbox = cal_bbox(visible_joints)
            cropped_image = image[bbox[2] : bbox[3], bbox[0] : bbox[1]]
            new_keypoint_list = []
            for i in range(pts.shape[0]):
                if pts[i][0] != 0 and pts[i][1] != 0:
                    new_keypoint = [pts[i][0] - bbox[0], pts[i][1] - bbox[2]]
                else:
                    new_keypoint = [pts[i][0], pts[i][1]]
                new_keypoint_list.append(new_keypoint)

            new_keypoint_list = np.array(new_keypoint_list)
            new_keypoint_list[:, 0] = new_keypoint_list[:, 0] / cropped_image.shape[1]
            new_keypoint_list[:, 1] = new_keypoint_list[:, 1] / cropped_image.shape[0]

            source = os.path.join(root_path, folder, image_name)
            target = os.path.join(output_path, "images", image_name)
            cv2.imwrite(target, cropped_image)
            info_json["category"].append(class_id)
            info_json["file_name"].append(image_name)
            info_json["regression_label"].append(new_keypoint_list.tolist())

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
    convert_synthetic_pose(
        split="train",
        classes=train_classes,
        root_path=args.root_dir,
        output_path="./Synthetic_animal_pose",
    )
