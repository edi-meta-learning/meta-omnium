# -----------------------
# Imports
# -----------------------
import argparse
import pandas as pd
import json
import os


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


if __name__ == "__main__":
    args = parse_arguments()

    labels_path = os.path.join(args.root_dir, args.dataset, "labels.csv")
    metadata = pd.read_csv(labels_path)

    info_path = os.path.join(args.root_dir, args.dataset, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    # Add the metadata from labels.csv file into the info.json file
    info["file_name"] = metadata["FILE_NAME"].tolist()
    info["category"] = metadata["CATEGORY"].tolist()

    if not metadata["SUPER_CATEGORY"].isnull()[0]:
        info["super_category"] = metadata["SUPER_CATEGORY"].tolist()

    info["task_type"] = "classification"

    # fix typo from Meta-Album info files
    info["total_super_categories"] = info["total_super_categorie"]
    del info["total_super_categorie"]

    with open(info_path, "w") as f:
        json.dump(info, f)
