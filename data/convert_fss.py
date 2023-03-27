# -----------------------
# Imports
# -----------------------
import json
import os
import shutil
import cv2
import glob


if __name__ == "__main__":
    for split in ["Trn", "Val", "Test"]:
        file_name_list = []
        category_list = []
        segmentation_file_name_list = []
        os.mkdir("FSS_" + split)
        os.mkdir("FSS_" + split + "/images")

        # Use the class splits from HSNet paper
        with open("FSS_Splits/%s.txt" % split, "r") as f:
            categories = f.read().split("\n")[:-1]
        categories = sorted(categories)

        for category in categories:
            for seg_name in glob.glob(
                os.path.join("fewshot_data", "fewshot_data", category) + "/*.png"
            ):
                seg_name = os.path.basename(seg_name)
                source_seg = os.path.join(
                    "fewshot_data", "fewshot_data", category, seg_name
                )
                target_seg = os.path.join(
                    "FSS_" + str(split), "images", category + "_" + seg_name
                )
                mask = cv2.imread(source_seg, 0)
                mask[mask >= 128] = 255
                mask[mask < 128] = 0
                cv2.imwrite(target_seg, mask)
                segmentation_file_name_list.append(category + "_" + seg_name)

                if os.path.exists(
                    os.path.join(
                        "fewshot_data",
                        "fewshot_data",
                        category,
                        seg_name.replace(".png", ".jpg"),
                    )
                ):
                    img_name = seg_name.replace(".png", ".jpg")

                file_name_list.append(category + "_" + img_name)
                category_list.append(category)

                source_img = os.path.join(
                    "fewshot_data", "fewshot_data", category, img_name
                )
                target_img = os.path.join(
                    "FSS_" + str(split), "images", category + "_" + img_name
                )
                shutil.copy(source_img, target_img)

        # Create the dictionary with relevant information
        num_classes = len(categories)
        info_path = os.path.join("FSS_" + str(split), "info.json")
        info = {
            "dataset_name": "FSS" + str(split),
            "dataset_description": "FSS-"
            + str(split)
            + " - Few-Shot Segmentation Dataset",
            "task_type": "segmentation",
            "total_categories": str(num_classes),
            "total_super_categories": 0,
            "uniform_number_of_images_per_category": True,
            "minimum_images_per_category": 10,
            "median_images_per_category": 10,
            "maximum_images_per_category": 10,
            "has_super_categories": False,
            "file_name": file_name_list,
            "category": category_list,
            "segmentation_file_name": segmentation_file_name_list,
        }

        # Store the information as a json file
        with open(info_path, "w") as f:
            json.dump(info, f)
