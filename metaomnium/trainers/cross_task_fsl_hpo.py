# Our code builds on https://github.com/ihsaan-ullah/meta-album

import os
import sys

# In order to import modules from packages in the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import argparse
import json

import random
import time
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import optuna
from metaomnium.dataloaders.data_loader_cross_problem_fsl import (
    DataLoaderCrossProblem,
    create_datasets,
    create_datasets_task_type,
    process_labels,
    get_k_keypoints,
)
from metaomnium.models.finetuning import FineTuning
from metaomnium.models.proto_finetuning import ProtoFineTuning
from metaomnium.models.fsl_resnet import ResNet
from metaomnium.models.maml import MAML
from metaomnium.models.proto_maml import ProtoMAML
from metaomnium.models.meta_curvature import MetaCurvature
from metaomnium.models.protonet import PrototypicalNetwork
from metaomnium.models.ddrr import DDRR
from metaomnium.models.train_from_scratch import TrainFromScratch

from utils.configs import (
    FT_CONF,
    PROTO_FT_CONF,
    MAML_CONF,
    PROTO_MAML_CONF,
    METACURVATURE_CONF,
    MATCHING_CONF,
    PROTO_CONF,
    DDRR_CONF,
    TFS_CONF,
)
from utils.utils import *


# -----------------------
# Arguments
# -----------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--train_datasets",
        type=str,
        required=True,
        help="datasets that will be used for training. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--val_id_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for in-domain validation. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--val_od_datasets",
        type=str,
        required=False,
        default="",
        help="datasets that will be used for out-domain validation. "
        + "Multiple datasets must be separeted by comma, e.g., BRD,CRS,FLW.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "tfs",
            "finetuning",
            "proto_finetuning",
            "maml",
            "protomaml",
            "metacurvature",
            "protonet",
            "matchingnet",
            "ddrr",
        ],
        required=True,
        help="which model to use",
    )

    # Optional arguments

    # Train/valid episodes config
    parser.add_argument(
        "--n_way_train",
        type=int,
        default=5,
        help="number of ways for the support set of the training episodes, "
        + "if None, the episodes are any-way tasks. Default: 5.",
    )
    parser.add_argument(
        "--k_shot_train",
        type=int,
        default=None,
        help="number of shots for the support set of the training episodes, "
        + "if None, the episodes are any-shot tasks. Default: None.",
    )
    parser.add_argument(
        "--min_shots_train",
        type=int,
        default=1,
        help="minimum number of shots for the any-shot training tasks. "
        + "Default: 1.",
    )
    parser.add_argument(
        "--max_shots_train",
        type=int,
        default=5,
        help="minimum number of shots for the any-shot training tasks. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--query_size_train",
        type=int,
        default=5,
        help="number of images for the query set of the training episodes. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=None,
        help="size of minibatches for training only applies for flat batch "
        + "models. Default: None.",
    )

    # Test episodes config
    parser.add_argument(
        "--n_way_eval",
        type=int,
        default=None,
        help="number of ways for the support set of the testing episodes, "
        + "if None, the episodes are any-way tasks. Default: None.",
    )
    parser.add_argument(
        "--k_shot_eval",
        type=int,
        default=None,
        help="number of shots for the support set of the testing episodes, "
        + "if None, the episodes are any-shot tasks. Default: None.",
    )
    parser.add_argument(
        "--min_ways_eval",
        type=int,
        default=2,
        help="minimum number of ways for the any-way testing tasks. " + "Default: 2.",
    )
    parser.add_argument(
        "--max_ways_eval",
        type=int,
        default=20,
        help="maximum number of ways for the any-way testing tasks. " + "Default: 20.",
    )
    parser.add_argument(
        "--min_shots_eval",
        type=int,
        default=1,
        help="minimum number of shots for the any-shot testing tasks. " + "Default: 1.",
    )
    parser.add_argument(
        "--max_shots_eval",
        type=int,
        default=20,
        help="minimum number of shots for the any-shot testing tasks. "
        + "Default: 20.",
    )
    parser.add_argument(
        "--query_size_eval",
        type=int,
        default=5,
        help="number of images for the query set of the testing episodes. "
        + "Default: 5.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="size of minibatches for testing only applies for flat-batch "
        + "models. Default: None.",
    )

    # Model configs
    parser.add_argument(
        "--runs", type=int, default=1, help="number of runs to perform. Default: 1."
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        default=None,
        help="number of meta-training iterations. Default: None.",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=600,
        help="number of meta-valid/test iterations. Default: 600.",
    )
    parser.add_argument(
        "--val_after",
        type=int,
        default=2500,
        help="after how many episodes the meta-validation should be "
        + "performed. Default: 2500.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="random seed to use. Default: 1337."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="validate performance on meta-validation tasks. Default: True.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        help="backbone to use. Default: resnet18.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=False,
        help="whether to freeze the weights in the finetuning model of "
        + "earlier layers. Default: False.",
    )
    parser.add_argument(
        "--meta_batch_size",
        type=int,
        default=1,
        help="number of tasks to compute outer-update. Default: 1.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="learning rate for (meta-)optimizer. Default: None.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=None,
        help="number of weight updates per training set. Default: None.",
    )
    parser.add_argument(
        "--T_val",
        type=int,
        default=None,
        help="number of weight updates at validation time. Default: None.",
    )
    parser.add_argument(
        "--T_test",
        type=int,
        default=None,
        help="number of weight updates at test time. Default: None.",
    )
    parser.add_argument(
        "--base_lr",
        type=float,
        default=None,
        help="inner level learning rate: Default: None.",
    )
    parser.add_argument(
        "--test_lr",
        type=float,
        default=0.001,
        help="learning rate to use at meta-val/test time for finetuning. "
        + "Default: 0.001.",
    )
    parser.add_argument(
        "--test_opt",
        choices=["adam", "sgd"],
        default="adam",
        help="optimizer to use at meta-val/test time for finetuning. "
        + "Default: adam.",
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="size of the images. Default: 128."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="none",
        help="root directory of the data and stored models",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dbg",
        help="name of the experiment for easy access to the results",
    )

    parser.add_argument(
        "--segm_classes",
        type=int,
        default=2,
        help="number of classes for segmentation. Default: 2.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="number of samples to use for HPO. Default: 64.",
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="whether to use pretrained model. Default: False.",
    )

    args, unparsed = parser.parse_known_args()

    args.train_episodes_config = {
        "n_way": args.n_way_train,
        "min_ways": None,
        "max_ways": None,
        "k_shot": args.k_shot_train,
        "min_shots": args.min_shots_train,
        "max_shots": args.max_shots_train,
        "query_size": args.query_size_train,
    }
    args.test_episodes_config = {
        "n_way": args.n_way_eval,
        "min_ways": args.min_ways_eval,
        "max_ways": args.max_ways_eval,
        "k_shot": args.k_shot_eval,
        "min_shots": args.min_shots_eval,
        "max_shots": args.max_shots_eval,
        "query_size": args.query_size_eval,
    }
    args.backbone = args.backbone.lower()

    return args, unparsed


class CrossTaskFewShotLearningExperiment:
    def __init__(self, args, selected_configuration):
        self.args = args

        # overwrite the configurations
        # note that momentum is only used for SGD
        if self.args.model == "tfs":
            TFS_CONF["lr"] = selected_configuration["lr"]
            TFS_CONF["opt_fn"] = selected_configuration["opt_fn"]
            TFS_CONF["momentum"] = selected_configuration["momentum"]
        elif self.args.model == "finetuning":
            FT_CONF["lr"] = selected_configuration["lr"]
            self.args.test_lr = selected_configuration["lr"]
            FT_CONF["opt_fn"] = selected_configuration["opt_fn"]
            self.args.test_opt = selected_configuration["opt_fn"]
            FT_CONF["momentum"] = selected_configuration["momentum"]
        elif self.args.model == "proto_finetuning":
            PROTO_FT_CONF["lr"] = selected_configuration["lr"]
            self.args.test_lr = selected_configuration["lr"]
            PROTO_FT_CONF["opt_fn"] = selected_configuration["opt_fn"]
            self.args.test_opt = selected_configuration["opt_fn"]
            PROTO_FT_CONF["momentum"] = selected_configuration["momentum"]
            PROTO_FT_CONF["init_lambda"] = selected_configuration["init_lambda"]
        elif self.args.model == "maml":
            MAML_CONF["lr"] = selected_configuration["lr"]
            MAML_CONF["opt_fn"] = selected_configuration["opt_fn"]
            MAML_CONF["momentum"] = selected_configuration["momentum"]
            MAML_CONF["base_lr"] = selected_configuration["base_lr"]
        elif self.args.model == "protomaml":
            PROTO_MAML_CONF["lr"] = selected_configuration["lr"]
            PROTO_MAML_CONF["opt_fn"] = selected_configuration["opt_fn"]
            PROTO_MAML_CONF["momentum"] = selected_configuration["momentum"]
            PROTO_MAML_CONF["init_lambda"] = selected_configuration["init_lambda"]
            PROTO_MAML_CONF["base_lr"] = selected_configuration["base_lr"]
        elif self.args.model == "metacurvature":
            METACURVATURE_CONF["lr"] = selected_configuration["lr"]
            METACURVATURE_CONF["opt_fn"] = selected_configuration["opt_fn"]
            METACURVATURE_CONF["momentum"] = selected_configuration["momentum"]
            METACURVATURE_CONF["base_lr"] = selected_configuration["base_lr"]
        elif self.args.model == "protonet":
            PROTO_CONF["lr"] = selected_configuration["lr"]
            PROTO_CONF["opt_fn"] = selected_configuration["opt_fn"]
            PROTO_CONF["momentum"] = selected_configuration["momentum"]
            PROTO_CONF["dist_temperature"] = selected_configuration["dist_temperature"]
        elif self.args.model == "matchingnet":
            MATCHING_CONF["lr"] = selected_configuration["lr"]
            MATCHING_CONF["opt_fn"] = selected_configuration["opt_fn"]
            MATCHING_CONF["momentum"] = selected_configuration["momentum"]
        elif self.args.model == "ddrr":
            DDRR_CONF["lr"] = selected_configuration["lr"]
            DDRR_CONF["opt_fn"] = selected_configuration["opt_fn"]
            DDRR_CONF["momentum"] = selected_configuration["momentum"]
            DDRR_CONF["init_lambda"] = selected_configuration["init_lambda"]

        # Define paths
        self.curr_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        if self.args.root_dir != "none":
            self.main_dir = self.args.root_dir
        else:
            self.main_dir = self.curr_dir

        self.data_dir = os.path.join(self.main_dir, "data")

        # Initialization step
        self.set_seed()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.gpu_info = get_torch_gpu_environment()
        self.clprint = lambda text: print(text)
        self.clprint("\n".join(self.gpu_info))
        self.configure()

    def set_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def configure(self):
        # Mapping from model names to configurations
        mod_to_conf = {
            "tfs": (TrainFromScratch, deepcopy(TFS_CONF)),
            "finetuning": (FineTuning, deepcopy(FT_CONF)),
            "proto_finetuning": (ProtoFineTuning, deepcopy(PROTO_FT_CONF)),
            "maml": (MAML, deepcopy(MAML_CONF)),
            "protomaml": (ProtoMAML, deepcopy(PROTO_MAML_CONF)),
            "metacurvature": (MetaCurvature, deepcopy(METACURVATURE_CONF)),
            "protonet": (PrototypicalNetwork, deepcopy(PROTO_CONF)),
            "ddrr": (DDRR, deepcopy(DDRR_CONF)),
        }

        # Get model constructor and config for the specified algorithm
        self.model_constr, self.conf = mod_to_conf[self.args.model]

        # Set configurations
        self.overwrite_conf(self.conf, "train_batch_size")
        self.overwrite_conf(self.conf, "test_batch_size")
        self.overwrite_conf(self.conf, "T")
        self.overwrite_conf(self.conf, "lr")
        self.overwrite_conf(self.conf, "meta_batch_size")
        self.overwrite_conf(self.conf, "freeze")
        self.overwrite_conf(self.conf, "base_lr")

        if self.args.test_opt is not None or self.args.test_lr is not None:
            self.overwrite_conf(self.conf, "test_opt")
            self.overwrite_conf(self.conf, "test_lr")
        self.args.opt_fn = self.conf["opt_fn"]

        # Make sure argument 'val_after' is specified when 'validate'=True
        if self.args.validate:
            assert not self.args.val_after is None, (
                "Please specify "
                + "val_after (number of episodes after which to perform "
                + "validation)"
            )

        # If using multi-step maml, perform gradient clipping with -5, +5
        if "T" in self.conf:
            if self.conf["T"] > 1 and (
                self.args.model == "maml"
                or self.args.model == "protomaml"
                or self.args.model == "metacurvature"
            ):
                self.conf["grad_clip"] = 5
            else:
                self.conf["grad_clip"] = None

        if self.args.T_test is None:
            self.conf["T_test"] = self.conf["T"]
        else:
            self.conf["T_test"] = self.args.T_test

        if self.args.T_val is None:
            self.conf["T_val"] = self.conf["T"]
        else:
            self.conf["T_val"] = self.args.T_val

        self.conf["dev"] = self.device
        backbone_name = "resnet"
        num_blocks = int(self.args.backbone.split(backbone_name)[1])
        self.conf["baselearner_fn"] = ResNet

        self.args.test_episodes_config["model"] = self.args.model
        self.args.test_episodes_config["training"] = False
        self.args.train_episodes_config["model"] = self.args.model
        self.args.train_episodes_config["training"] = True

        num_tasks = self.args.eval_iters if self.args.val_id_datasets else 0
        val_id_datasets, val_id_dataset_task_type_dict = create_datasets(
            self.args.val_id_datasets.split(","), self.data_dir
        )
        self.val_id_loader = DataLoaderCrossProblem(
            val_id_datasets, num_tasks, self.args.test_episodes_config
        )

        num_tasks = self.args.eval_iters if self.args.val_od_datasets else 0
        val_od_datasets, val_od_dataset_task_type_dict = create_datasets(
            self.args.val_od_datasets.split(","), self.data_dir
        )
        self.val_od_loader = DataLoaderCrossProblem(
            val_od_datasets, num_tasks, self.args.test_episodes_config
        )

        num_train_classes = self.args.n_way_train
        if self.args.model in ("tfs", "finetuning", "proto_finetuning"):
            self.args.batchmode = True
            (
                train_datasets,
                train_dataset_task_type_dict,
                weights,
            ) = create_datasets_task_type(
                self.args.train_datasets.split(","), self.data_dir
            )
            if "finetuning" in self.args.model:
                # find the number of training classes for classification
                if "classification" in train_datasets:
                    num_train_classes = len(
                        train_datasets["classification"].idx_per_label
                    )
            if "segmentation" in train_datasets:
                # one class for background
                self.args.segm_classes = (
                    len(train_datasets["segmentation"].idx_per_label) + 1
                )
            train_datasets = list(train_datasets.values())
        else:
            self.args.batchmode = False
            train_datasets, train_dataset_task_type_dict = create_datasets(
                self.args.train_datasets.split(","), self.data_dir
            )
            weights = None
        self.train_loader = DataLoaderCrossProblem(
            train_datasets,
            self.args.train_iters,
            self.args.train_episodes_config,
            weights=weights,
        )

        self.dataset_task_type_dict = {}
        for dataset in train_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = train_dataset_task_type_dict[dataset]
        for dataset in val_id_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = val_id_dataset_task_type_dict[
                dataset
            ]
        for dataset in val_od_dataset_task_type_dict:
            self.dataset_task_type_dict[dataset] = val_od_dataset_task_type_dict[
                dataset
            ]

        self.conf["baselearner_args"] = {
            "num_blocks": num_blocks,
            "dev": self.device,
            "train_classes": num_train_classes,
            "criterion": nn.CrossEntropyLoss(),
            "img_size": self.args.img_size,
            "segm_classes": self.args.segm_classes,
            "pretrained": self.args.pretrained,
        }

        # Print the configuration for confirmation
        self.clprint("\n\n### ------------------------------------------ ###")
        self.clprint(f"Model: {self.args.model}")
        self.clprint(f"Training Datasets: {self.args.train_datasets}")
        self.clprint(f"In-Domain Validation Datasets: {self.args.val_id_datasets}")
        self.clprint(f"Out-Domain Validation Datasets: {self.args.val_od_datasets}")
        self.clprint(f"Random Seed: {self.args.seed}")
        self.print_conf()
        self.clprint("### ------------------------------------------ ###\n")

    def overwrite_conf(self, conf, arg_str):
        # If value provided in arguments, overwrite the config with it
        value = getattr(self.args, arg_str)
        if value is not None:
            conf[arg_str] = value
        else:
            if arg_str not in conf:
                conf[arg_str] = None
            else:
                setattr(self.args, arg_str, conf[arg_str])

    def cycle(self, iterable):
        while True:
            for x in iterable:
                yield x

    def print_conf(self):
        self.clprint(f"Configuration dump:")
        for k in self.conf.keys():
            self.clprint(f"\t{k} : {self.conf[k]}")

    def validate(self, model, val_loader):
        # We will store the statistics for each dataset separately
        input_size_dict = {}
        running_loss_dict = {}
        running_corrects_dict = {}

        for i, task in enumerate(val_loader):
            n_way = task.n_way
            k_shot = task.k_shot
            query_size = task.query_size
            data = task.data
            labels = task.labels
            support_size = n_way * k_shot

            if task.task_type == "segmentation":
                labels = task.segmentations.squeeze(dim=1)
            elif task.task_type.startswith("regression"):
                if task.task_type in [
                    "regression_pose_animals",
                    "regression_pose_animals_syn",
                    "regression_mpii",
                ]:
                    labels = get_k_keypoints(
                        n_way * 5, task.regressions, task.task_type
                    )
                else:
                    labels = task.regressions
            else:
                labels = process_labels(n_way * (k_shot + query_size), n_way)

            train_x, train_y, test_x, test_y = (
                data[:support_size],
                labels[:support_size],
                data[support_size:],
                labels[support_size:],
            )
            acc, loss_history, _, _ = model.evaluate(
                n_way, train_x, train_y, test_x, test_y, task_type=task.task_type
            )
            loss = loss_history[-1]
            labels = test_y.cpu().numpy()
            curr_input_size = len(test_y)

            if task.dataset not in input_size_dict:
                input_size_dict[task.dataset] = 0
            if task.dataset not in running_loss_dict:
                running_loss_dict[task.dataset] = 0
            if task.dataset not in running_corrects_dict:
                running_corrects_dict[task.dataset] = 0
            input_size_dict[task.dataset] += curr_input_size
            running_loss_dict[task.dataset] += loss * curr_input_size
            running_corrects_dict[task.dataset] += acc * curr_input_size

            if (i + 1) == self.args.eval_iters:
                break

        val_loss_dict = {
            e: running_loss_dict[e] / input_size_dict[e] for e in running_loss_dict
        }
        val_score_dict = {
            e: running_corrects_dict[e] / input_size_dict[e]
            for e in running_corrects_dict
        }
        val_error_dict = {}
        for dataset_name in val_score_dict:
            # The following regression benchmarks report the error already
            if (
                ("distractor" in dataset_name)
                or ("Pascal1D" in dataset_name)
                or ("ShapeNet1D" in dataset_name)
                or ("ShapeNet2D" in dataset_name)
            ):
                val_error_dict[dataset_name] = val_score_dict[dataset_name]
            # Convert the scores for other benchmarks to error
            else:
                val_error_dict[dataset_name] = 1.0 - val_score_dict[dataset_name]

        val_loss = np.mean([val_loss_dict[e] for e in val_loss_dict])
        val_score = np.mean([val_score_dict[e] for e in val_score_dict])

        return val_error_dict, val_score, val_loss, val_score_dict, val_loss_dict

    def run(self):
        seeds = [random.randint(0, 100000) for _ in range(self.args.runs)]
        self.clprint(f"Run seeds: {seeds}")

        for run in range(self.args.runs):
            self.clprint("\n\n" + "-" * 40)
            self.clprint(f"[*] Starting run {run} with seed {seeds[run]}")

            torch.manual_seed(seeds[run])
            model = self.model_constr(**self.conf)

            train_generator = iter(self.train_loader.generator(seeds[run]))
            val_id_generator = iter(self.val_id_loader.generator(seeds[run]))
            val_od_generator = iter(self.val_od_loader.generator(seeds[run]))

            if model.trainable:
                for i, task in enumerate(train_generator):

                    n_way = task.n_way
                    k_shot = task.k_shot
                    query_size = task.query_size
                    data = task.data
                    labels = task.labels
                    support_size = n_way * k_shot

                    if self.args.batchmode:
                        train_x = data
                        # Process the labels according to the task type
                        if task.task_type == "segmentation":
                            train_y = task.segmentations.squeeze(dim=1) * (
                                task.labels + 1
                            ).unsqueeze(dim=1).unsqueeze(dim=2)
                        elif task.task_type.startswith("regression"):
                            if task.task_type in [
                                "regression_pose_animals",
                                "regression_pose_animals_syn",
                                "regression_mpii",
                            ]:
                                train_y = get_k_keypoints(
                                    n_way * 5, task.regressions, task.task_type
                                )

                            else:
                                train_y = task.regressions
                        elif task.task_type == "classification":
                            train_y = labels.view(-1)
                        train_score, train_loss, _, _ = model.train(
                            train_x, train_y, task_type=task.task_type
                        )
                    else:
                        # Process the labels according to the task type
                        if task.task_type == "segmentation":
                            labels = task.segmentations.squeeze(dim=1)
                        elif task.task_type.startswith("regression"):
                            if task.task_type in [
                                "regression_pose_animals",
                                "regression_pose_animals_syn",
                                "regression_mpii",
                            ]:
                                labels = get_k_keypoints(
                                    n_way * 5, task.regressions, task.task_type
                                )

                            else:
                                labels = task.regressions
                        else:
                            labels = process_labels(
                                n_way * (k_shot + query_size), n_way
                            )

                        train_x, train_y, test_x, test_y = (
                            data[:support_size],
                            labels[:support_size],
                            data[support_size:],
                            labels[support_size:],
                        )

                        train_score, train_loss, _, _ = model.train(
                            train_x, train_y, test_x, test_y, task_type=task.task_type
                        )

                    if (i + 1) == self.args.train_iters:
                        break

        (
            id_error_dict,
            val_id_score,
            val_id_loss,
            val_id_scores,
            val_id_losses,
        ) = self.validate(model, val_id_generator)
        (
            od_error_dict,
            val_od_score,
            val_od_loss,
            val_od_scores,
            val_od_losses,
        ) = self.validate(model, val_od_generator)

        return {
            "val_id_errors": id_error_dict,
            "val_od_errors": od_error_dict,
            "val_id_losses": val_id_losses,
            "val_od_losses": val_od_losses,
        }


def sample_configuration(trial, model):
    if model == "tfs":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
        }
    elif model == "finetuning":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
        }
    elif model == "proto_finetuning":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "init_lambda": trial.suggest_float("init_lambda", 1e-2, 100.0, log=True),
        }
    elif model == "maml":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "base_lr": trial.suggest_float("base_lr", 1e-3, 0.5, log=True),
        }
    elif model == "protomaml":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "base_lr": trial.suggest_float("base_lr", 1e-3, 0.5, log=True),
            "init_lambda": trial.suggest_float("init_lambda", 1e-2, 100.0, log=True),
        }
    elif model == "metacurvature":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "base_lr": trial.suggest_float("base_lr", 1e-3, 0.5, log=True),
        }
    elif model == "protonet":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "dist_temperature": trial.suggest_float(
                "dist_temperature", 0.1, 10.0, log=True
            ),
        }
    elif model == "matchingnet":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
        }
    elif model == "ddrr":
        return {
            "lr": trial.suggest_float("lr", 1e-4, 0.1, log=True),
            "opt_fn": trial.suggest_categorical("opt_fn", ["adam", "sgd"]),
            "momentum": trial.suggest_categorical("momentum", [0.0, 0.9, 0.99]),
            "init_lambda": trial.suggest_float("init_lambda", 1e-2, 100.0, log=True),
        }


def objective(trial, args):
    current_trial = trial.number
    print("Current trial: " + str(current_trial))

    # It can sometimes happen that a trial fails
    # e.g. due to configuration that leads to extreme behaviour.
    try:
        config = sample_configuration(trial, args.model)
        experiment = CrossTaskFewShotLearningExperiment(args, config)
        scores = experiment.run()
        processed_scores = []
        for val_id_dataset in args.val_id_datasets.split(","):
            if str(scores["val_id_errors"][val_id_dataset]) == "nan":
                processed_scores.append(9999.0)
            else:
                processed_scores.append(scores["val_id_errors"][val_id_dataset])
        for val_od_dataset in args.val_od_datasets.split(","):
            if str(scores["val_od_errors"][val_od_dataset]) == "nan":
                processed_scores.append(9999.0)
            else:
                processed_scores.append(scores["val_od_errors"][val_od_dataset])
    except:
        print("Trial #{} failed: ".format(current_trial))
        processed_scores = [9999.0] * (
            len(args.val_id_datasets.split(",")) + len(args.val_od_datasets.split(","))
        )

    return processed_scores


if __name__ == "__main__":
    # Get args
    args, unparsed = parse_arguments()

    # If there is still some unparsed argument, raise error
    if len(unparsed) != 0:
        raise ValueError(f"Argument {unparsed} not recognized")

    start_time = time.time()

    metrics = []
    for val_id_dataset in args.val_id_datasets.split(","):
        metrics.append(("val_id_errors", val_id_dataset))
    for val_od_dataset in args.val_od_datasets.split(","):
        metrics.append(("val_od_errors", val_id_dataset))

    # We minimize all of the errors
    modes = ["minimize" for _ in metrics]

    # Ensure reproducible sampling of configurations
    random.seed(args.seed)
    np.random.seed(args.seed)
    objective_func = lambda trial: objective(trial, args)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(directions=modes, sampler=sampler)
    study.optimize(objective_func, n_trials=args.num_samples)

    results = study.trials

    sampled_configurations = [results[i].params for i in range(len(results))]
    scores_per_configuration = [results[i].values for i in range(len(results))]

    # Process the results
    all_details = []
    for config, scores in zip(sampled_configurations, scores_per_configuration):
        try:
            details = {}
            for metric, score in zip(metrics, scores):
                details["metric/" + metric[0] + "/" + metric[1]] = score
            for key_cfg in config:
                details["config/" + key_cfg] = config[key_cfg]
            all_details.append(details)
        except:
            print("Error with config: " + str(config))
            print("Scores: " + str(scores))

    hpo_time = time.time() - start_time
    print("Time needed for HPO: " + str(hpo_time) + " seconds")

    if len(all_details) == 0:
        summary_dict = {
            "hpo_time": hpo_time,
            "model": args.model,
            "train_datasets": args.train_datasets,
            "val_id_datasets": args.val_id_datasets,
            "val_od_datasets": args.val_od_datasets,
            "num_samples": args.num_samples,
            "sampled_configurations": sampled_configurations,
            "scores_per_configuration": scores_per_configuration,
        }
        print("No valid configuration found")
    else:
        # Create the dataframe that we can analyse
        results_df = pd.DataFrame(all_details)

        result_columns = [
            e for e in results_df.columns if "metric" in e and "errors" in e
        ]
        lowest_errors = results_df[result_columns].min(axis=0)
        normalized_errors = results_df[result_columns] / lowest_errors
        mean_normalized_errors = normalized_errors.mean(axis=1)
        best_configuration_idx = mean_normalized_errors.argmin(axis=0)
        best_configuration = results_df.loc[best_configuration_idx]

        # Store the results of the best configuration
        best_configuration_results = {
            "val_id_errors": {},
            "val_od_errors": {},
            "val_id_losses": {},
            "val_od_losses": {},
        }
        result_columns = [e for e in results_df.columns if "metric" in e]
        for result_name in result_columns:
            split_name = result_name.split("/")[-2]
            dataset_name = result_name.split("/")[-1]
            best_configuration_results[split_name][dataset_name] = best_configuration[
                result_name
            ]

        # Store best hyperparameters as a dictionary
        hp_columns = [e for e in results_df.columns if "config" in e]
        best_hyperparameters = best_configuration[hp_columns]
        best_hyperparameters_values = {}
        for hp_name in hp_columns:
            hp_name_short = hp_name.split("/")[1]
            best_hyperparameters_values[hp_name_short] = best_hyperparameters[hp_name]

        hpo_time = time.time() - start_time

        summary_dict = {
            "hpo_time": hpo_time,
            "model": args.model,
            "train_datasets": args.train_datasets,
            "val_id_datasets": args.val_id_datasets,
            "val_od_datasets": args.val_od_datasets,
            "num_samples": args.num_samples,
            "best_configuration_results": best_configuration_results,
            "best_hyperparameters": best_hyperparameters_values,
            "sampled_configurations": sampled_configurations,
            "scores_per_configuration": scores_per_configuration,
        }

    hpo_summaries_directory = "hpo_summaries"
    if not os.path.exists(hpo_summaries_directory):
        os.makedirs(hpo_summaries_directory)
    with open(
        os.path.join(hpo_summaries_directory, args.experiment_name + ".json"), "w"
    ) as f:
        json.dump(summary_dict, f)
