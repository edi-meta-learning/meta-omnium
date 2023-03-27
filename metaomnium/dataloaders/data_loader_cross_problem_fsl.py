import json
import os.path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.utils import check_random_state
from torch.utils.data import Dataset
from torchvision import transforms


class ClassificationDataset(Dataset):
    def __init__(self, datasets, data_dir, img_size=128):
        self.task_type = "classification"
        if len(datasets) == 1:
            self.name = datasets[0]
        else:
            self.name = f"Multiple datasets: {','.join(datasets)}"
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

        self.img_paths = list()
        self.labels = list()
        id_ = 0
        for dataset in datasets:
            with open(os.path.join(self.data_dir, dataset, "info.json"), "r") as f:
                info = json.load(f)
            img_path = f"{self.data_dir}/{dataset}/images/"
            self.img_paths.extend(
                [os.path.join(img_path, x) for x in info["file_name"]]
            )

            # Transform string labels into non-negative integer IDs
            label_to_id = dict()
            for label in info["category"]:
                if label not in label_to_id:
                    label_to_id[label] = id_
                    id_ += 1
                self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)

        self.idx_per_label = []
        for i in range(max(self.labels) + 1):
            idx = np.argwhere(self.labels == i).reshape(-1)
            self.idx_per_label.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()


class SegmentationDataset(Dataset):
    def __init__(self, datasets, data_dir, img_size=224):
        self.task_type = "segmentation"
        if len(datasets) == 1:
            self.name = datasets[0]
        else:
            self.name = f"Multiple datasets: {','.join(datasets)}"
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

        # We will remember paths to both images and segmentation masks
        self.img_paths = list()
        self.labels = list()
        self.segm_paths = list()

        id_ = 0
        for dataset in datasets:
            with open(os.path.join(self.data_dir, dataset, "info.json"), "r") as f:
                info = json.load(f)
            img_path = f"{self.data_dir}/{dataset}/images/"
            self.img_paths.extend(
                [os.path.join(img_path, x) for x in info["file_name"]]
            )
            self.segm_paths.extend(
                [os.path.join(img_path, x) for x in info["segmentation_file_name"]]
            )

            # Transform string labels into non-negative integer IDs
            label_to_id = dict()
            for label in info["category"]:
                if label not in label_to_id:
                    label_to_id[label] = id_
                    id_ += 1
                self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)

        self.idx_per_label = []
        for i in range(max(self.labels) + 1):
            idx = np.argwhere(self.labels == i).reshape(-1)
            self.idx_per_label.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label, segm_path = self.img_paths[i], self.labels[i], self.segm_paths[i]
        image = self.transform(Image.open(path))
        segm_image = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        segm_image = self.mask_transform(Image.fromarray(segm_image))
        return image, torch.LongTensor([label]).squeeze(), segm_image.long()


# RegressionDataset is shared between keypoint and regression tasks
class RegressionDataset(Dataset):
    def __init__(self, datasets, data_dir, img_size=128, task_type=None):
        self.task_type = task_type
        if len(datasets) == 1:
            self.name = datasets[0]
        else:
            self.name = f"Multiple datasets: {','.join(datasets)}"
        self.data_dir = data_dir

        self.img_size = img_size
        self.transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )

        self.img_paths = list()
        self.labels = list()
        self.regressions = list()
        id_ = 0
        for dataset in datasets:
            with open(os.path.join(self.data_dir, dataset, "info.json"), "r") as f:
                info = json.load(f)
            img_path = f"{self.data_dir}/{dataset}/images/"
            self.img_paths.extend(
                [os.path.join(img_path, x) for x in info["file_name"]]
            )
            if self.task_type in ["regression_distractor"]:
                # Only keep [x,y]
                self.regressions = [
                    [(float(j[0]) / img_size), float(j[1]) / img_size]
                    for j in info["center"]
                ]
            elif self.task_type in ["regression_pascal1d"]:
                labels = info["regression_label"]
                for y in labels:
                    self.regressions.append(y)
            elif self.task_type in ["regression_shapenet1d"]:
                # Transform y degree to [cos(y), sin(y), y]
                y_degree = info["regression_label"]
                regression_label_processed = list()
                for y in y_degree:
                    y = y * 2 * np.pi
                    regression_label_processed.append([np.cos(y), np.sin(y), y])
                self.regressions.extend(regression_label_processed)
            elif self.task_type in ["regression_shapenet2d"]:
                y_list = info["regression_label"]
                regression_label_processed = list()
                for y in y_list:
                    gt = [float(i) for i in y.split("_")]
                    regression_label_processed.append(gt)
                self.regressions.extend(regression_label_processed)
            else:
                self.regressions = info["regression_label"]

            label_to_id = dict()
            for label in info["category"]:
                if label not in label_to_id:
                    label_to_id[label] = id_
                    id_ += 1
                self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)

        self.idx_per_label = []
        for i in range(max(self.labels) + 1):
            idx = np.argwhere(self.labels == i).reshape(-1)
            self.idx_per_label.append(idx)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        path, label, regression = self.img_paths[i], self.labels[i], self.regressions[i]
        image = self.transform(Image.open(path).convert("RGB"))
        category = torch.LongTensor([label]).squeeze()
        if self.task_type in ["regression_pascal1d"]:
            norm_degree = regression
            regression = [norm_degree]
            regression_label = torch.FloatTensor(regression)
        elif self.task_type in ["regression_pose_animals_syn", "regression_mpii"]:
            regression_label = torch.FloatTensor([regression]).squeeze()
        elif self.task_type in ["regression_shapenet1d"]:
            regression_label = torch.FloatTensor(regression)
        elif self.task_type in ["regression_distractor", "regression_shapenet2d"]:
            regression_label = torch.FloatTensor(regression)
        elif self.task_type in ["regression_pose_animals"]:
            # The keypoints need to be scaled as we resized the image
            orig_image_shape = np.asarray(Image.open(path).convert("RGB")).shape
            regression = np.array(regression)
            regression_label = np.zeros(regression.shape)
            regression_label[:, 0] = regression[:, 0] / orig_image_shape[1]
            regression_label[:, 1] = regression[:, 1] / orig_image_shape[0]
            regression_label = torch.FloatTensor(regression_label)
            zeros = torch.zeros_like(regression_label)
            regression_label = torch.where(
                regression_label > 1.0, zeros, regression_label
            )
        return image, category, regression_label


def process_labels(batch_size, num_classes):
    return torch.arange(num_classes).repeat(batch_size // num_classes).long()


def get_k_keypoints(num_ways, keypoints, task_type):
    if task_type == "regression_pose_animals":
        indices = np.random.choice(keypoints.shape[1], num_ways, replace=False)
        keypoints = keypoints[:, indices, :]
    else:
        indices = torch.randperm(keypoints.size()[1])
        keypoints = keypoints[:, indices, :]

    return keypoints.view(keypoints.shape[0], -1)


def create_datasets(datasets, data_dir):
    torch_datasets = []
    dataset_task_type_dict = {}
    for dataset in datasets:
        if dataset:
            # find the task type
            with open(os.path.join(data_dir, dataset, "info.json"), "r") as f:
                info = json.load(f)
            task_type = info["task_type"]

            if task_type == "classification":
                torch_dataset = ClassificationDataset([dataset], data_dir)
                torch_datasets.append(torch_dataset)
            elif task_type == "segmentation":
                torch_dataset = SegmentationDataset([dataset], data_dir)
                torch_datasets.append(torch_dataset)
            elif task_type.startswith("regression"):
                torch_dataset = RegressionDataset(
                    [dataset], data_dir, task_type=task_type
                )
                torch_datasets.append(torch_dataset)
            dataset_task_type_dict[dataset] = task_type

    return torch_datasets, dataset_task_type_dict


def create_datasets_task_type(datasets, data_dir):
    torch_datasets = {}
    dataset_task_type_dict = {}
    # identify which datasets go together based on task type
    datasets_per_task_type = {}
    total_num_datasets = len(datasets)
    for dataset in datasets:
        # find the task type
        with open(os.path.join(data_dir, dataset, "info.json"), "r") as f:
            info = json.load(f)
        task_type = info["task_type"]
        if task_type not in datasets_per_task_type:
            datasets_per_task_type[task_type] = []
        datasets_per_task_type[task_type].append(dataset)

    weights = []
    for task_type in datasets_per_task_type:
        # find the task type and create appropriate dataset that combines
        # multiple actual datasets of the same type
        if task_type == "classification":
            torch_dataset = ClassificationDataset(
                datasets_per_task_type[task_type], data_dir
            )
        elif task_type == "segmentation":
            torch_dataset = SegmentationDataset(
                datasets_per_task_type[task_type], data_dir
            )
        elif task_type.startswith("regression"):
            torch_dataset = RegressionDataset(
                datasets_per_task_type[task_type], data_dir, task_type=task_type
            )
        torch_datasets[task_type] = torch_dataset
        dataset_task_type_dict[task_type] = task_type
        weights.append(len(datasets_per_task_type[task_type]) / total_num_datasets)

    return torch_datasets, dataset_task_type_dict, weights


class ClassificationTask:
    def __init__(self, n_way, k_shot, query_size, data, labels, dataset, task_type):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.data = data
        self.labels = labels
        self.dataset = dataset
        self.task_type = task_type


class SegmentationTask:
    def __init__(
        self, n_way, k_shot, query_size, data, labels, segmentations, dataset, task_type
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.data = data
        self.labels = labels
        self.segmentations = segmentations
        self.dataset = dataset
        self.task_type = task_type


# RegressionTask is shared between keypoint estimation and regression
class RegressionTask:
    def __init__(
        self, n_way, k_shot, query_size, data, labels, regressions, dataset, task_type
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.data = data
        self.labels = labels
        self.dataset = dataset
        self.task_type = task_type
        self.regressions = regressions


class DataLoaderCrossProblem:
    def __init__(
        self, datasets, num_tasks, episodes_config, test_loader=False, weights=None
    ):
        self.datasets = datasets
        self.num_tasks = num_tasks
        self.test_loader = test_loader
        self.n_way = episodes_config["n_way"]
        self.min_ways = episodes_config["min_ways"]
        self.max_ways = episodes_config["max_ways"]
        self.k_shot = episodes_config["k_shot"]
        self.min_shots = episodes_config["min_shots"]
        self.max_shots = episodes_config["max_shots"]
        self.query_size = episodes_config["query_size"]
        self.model = episodes_config["model"]
        self.training = episodes_config["training"]
        # make the distribution of problems across tasks
        # same during batch-mode and standard few-shot learning
        self.weights = weights

    def generator(self, seed):
        if not self.test_loader:
            while True:
                random_gen = check_random_state(seed)
                if self.num_tasks == 0:
                    break
                for _ in range(self.num_tasks):
                    # Select task configuration
                    dataset_idx = random_gen.choice(
                        range(len(self.datasets)), p=self.weights
                    )
                    dataset = self.datasets[dataset_idx]

                    idx_per_label = dataset.idx_per_label
                    num_classes = len(idx_per_label)
                    n_way, k_shot = self.prepare_task_config(
                        num_classes, random_gen, dataset
                    )

                    if (
                        dataset.task_type == "regression_pascal1d"
                        or dataset.task_type == "regression_shapenet1d"
                        or dataset.task_type == "regression_distractor"
                        or dataset.task_type == "regression_shapenet2d"
                    ):
                        k_shot = 5 * k_shot

                    # Select examples for the task
                    batch = list()
                    if self.training and self.model == "finetuning":
                        total_examples = n_way * (k_shot + self.query_size)
                        selected_idx = random_gen.choice(
                            range(len(dataset)), total_examples, replace=False
                        )
                        batch.append(selected_idx)
                    else:
                        total_examples = k_shot + self.query_size
                        classes = random_gen.permutation(num_classes)[:n_way]
                        for c in classes:
                            idx = idx_per_label[c]
                            selected_idx = random_gen.choice(
                                idx, total_examples, replace=False
                            )
                            batch.append(selected_idx)
                    batch = np.stack(batch).T.reshape(-1)

                    # Load the examples
                    if dataset.task_type == "classification":
                        data = list()
                        labels = list()
                        for i in batch:
                            img, label = dataset[i]
                            data.append(img)
                            labels.append(label)
                        data = torch.stack(data)
                        labels = torch.stack(labels)

                        # Return the task
                        task = ClassificationTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            dataset.name,
                            dataset.task_type,
                        )
                    elif dataset.task_type == "segmentation":
                        data = list()
                        labels = list()
                        segmentations = list()
                        for i in batch:
                            img, label, segmentation = dataset[i]
                            data.append(img)
                            labels.append(label)
                            segmentations.append(segmentation)
                        data = torch.stack(data)
                        labels = torch.stack(labels)
                        segmentations = torch.stack(segmentations)

                        # Return the task
                        task = SegmentationTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            segmentations,
                            dataset.name,
                            dataset.task_type,
                        )
                    elif dataset.task_type.startswith("regression"):
                        data = list()
                        labels = list()
                        regressions = list()
                        for i in batch:
                            img, label, regression = dataset[i]
                            data.append(img)
                            labels.append(label)
                            regressions.append(regression)
                        data = torch.stack(data)
                        labels = torch.stack(labels)
                        regressions = torch.stack(regressions)
                        # Return the task
                        task = RegressionTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            regressions,
                            dataset.name,
                            dataset.task_type,
                        )
                    yield task
        else:
            for dataset_idx in range(len(self.datasets)):
                random_gen = check_random_state(seed)

                # Dataset information
                dataset = self.datasets[dataset_idx]
                idx_per_label = dataset.idx_per_label
                num_classes = len(idx_per_label)
                for _ in range(self.num_tasks):
                    # Select task configuration
                    n_way, k_shot = self.prepare_task_config(
                        num_classes, random_gen, dataset
                    )

                    if (
                        dataset.task_type == "regression_pascal1d"
                        or dataset.task_type == "regression_shapenet1d"
                        or dataset.task_type == "regression_distractor"
                        or dataset.task_type == "regression_shapenet2d"
                    ):
                        k_shot = 5 * k_shot

                    total_examples = k_shot + self.query_size

                    # Select examples for the task
                    batch = list()
                    classes = random_gen.permutation(num_classes)[:n_way]
                    for c in classes:
                        idx = idx_per_label[c]
                        selected_idx = random_gen.choice(
                            idx, total_examples, replace=False
                        )
                        batch.append(selected_idx)
                    batch = np.stack(batch).T.reshape(-1)

                    # Load the examples
                    if dataset.task_type == "classification":
                        data = list()
                        labels = list()
                        for i in batch:
                            img, label = dataset[i]
                            data.append(img)
                            labels.append(label)
                        data = torch.stack(data)
                        labels = torch.stack(labels)

                        # Return the task
                        task = ClassificationTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            dataset.name,
                            dataset.task_type,
                        )

                    elif dataset.task_type == "segmentation":
                        data = list()
                        labels = list()
                        segmentations = list()
                        for i in batch:
                            img, label, segmentation = dataset[i]
                            data.append(img)
                            labels.append(label)
                            segmentations.append(segmentation)
                        data = torch.stack(data)
                        labels = torch.stack(labels)
                        segmentations = torch.stack(segmentations)

                        # Return the task
                        task = SegmentationTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            segmentations,
                            dataset.name,
                            dataset.task_type,
                        )
                    elif dataset.task_type.startswith("regression"):
                        data = list()
                        labels = list()
                        regressions = list()
                        for i in batch:
                            img, label, regression = dataset[i]
                            data.append(img)
                            labels.append(label)
                            regressions.append(regression)
                        data = torch.stack(data)
                        labels = torch.stack(labels)
                        regressions = torch.stack(regressions)
                        # Return the task
                        task = RegressionTask(
                            n_way,
                            k_shot,
                            self.query_size,
                            data,
                            labels,
                            regressions,
                            dataset.name,
                            dataset.task_type,
                        )

                    yield task

    def prepare_task_config(self, num_classes, random_gen, dataset):
        if dataset.task_type == "segmentation":
            n_way = 1
        elif dataset.task_type.startswith("regression"):
            n_way = 1
        else:
            n_way = self.n_way

        if n_way is None:
            max_ways = num_classes
            if self.max_ways < max_ways:
                max_ways = self.max_ways

            n_way = random_gen.randint(self.min_ways, max_ways + 1)

        k_shot = self.k_shot
        if k_shot is None:
            k_shot = random_gen.randint(self.min_shots, self.max_shots + 1)

        return n_way, k_shot
