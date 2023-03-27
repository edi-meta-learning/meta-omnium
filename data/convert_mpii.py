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


def convert_mpii(split, root_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))

    info_f = open(os.path.join(output_path, "info.json"), "w")
    info_json = dict()
    info_json["file_name"] = list()
    info_json["category"] = list()
    info_json["regression_label"] = list()
    info_json["task_type"] = "regression_mpii"
    info_json["total_categories"] = 1
    info_json["minimum_images_per_category"] = 72
    info_json["median_images_per_category"] = 72
    info_json["maximum_images_per_category"] = 72
    info_json["num_keypoints"] = 16

    target_dir = os.path.join(output_path, "images")
    annot = json.load(open(os.path.join(root_path, "annot", "trainval.json")))
    for i in range(0, len(annot)):
        data = annot[i]
        joints = np.array(data["joints"])
        image = data["image"]
        image_new = str(i) + ".jpg"

        indices = np.where(np.all([joints[:, 0] > 0, joints[:, 1] > 0], axis=0))

        visible_joints = joints[indices].reshape(-1, 2)
        if visible_joints.shape[0] != 0:
            bbox = cal_bbox(visible_joints)
            image_path = os.path.join(args.root_dir, "images", image)
            image_np = cv2.imread(image_path)
            cropped_image = image_np[bbox[2] : bbox[3], bbox[0] + 5 : bbox[1] + 5]

            # Receneter points according to min x coordinate of bouding box
            # print("prev", joints)
            if cropped_image.shape[0] != 0 and cropped_image.shape[1] != 0:
                print(image_new)
                joints[:, 0] -= bbox[0]
                joints[:, 1] -= bbox[2]
                joints[:, 0] /= cropped_image.shape[1]
                joints[:, 1] /= cropped_image.shape[0]
                joints[joints < 0] = 0.0
                # Crop image using bounding box
                cv2.imwrite(os.path.join(target_dir, image_new), cropped_image)
                info_json["file_name"].append(image_new)
                info_json["category"].append(0)
                info_json["regression_label"].append(joints.tolist())

    print(info_json)

    b = json.dumps(info_json)
    info_f.write(b)
    info_f.close()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)

    num_classes = 5

    convert_mpii(split="train", root_path=args.root_dir, output_path="./MPII-1")
