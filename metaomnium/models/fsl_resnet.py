import math
from collections import OrderedDict
from itertools import accumulate

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, dev):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dev = dev

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, momentum=1)
        self.skip = stride > 1
        if self.skip:
            self.conv3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.bn3 = nn.BatchNorm2d(num_features=out_channels, momentum=1)

    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu(z)
        z = self.conv2(z)
        z = self.bn2(z)

        y = x
        if self.skip:
            y = self.conv3(y)
            y = self.bn3(y)
        return self.relu(y + z)

    def forward_weights(self, x, weights):
        z = F.conv2d(
            input=x,
            weight=weights[0],
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )

        z = F.batch_norm(
            z,
            torch.zeros(self.bn1.running_mean.size()).to(self.dev),
            torch.ones(self.bn1.running_var.size()).to(self.dev),
            weights[1],
            weights[2],
            momentum=1,
            training=True,
        )

        z = F.relu(z)

        z = F.conv2d(
            input=z, weight=weights[3], bias=None, stride=1, padding=self.padding
        )

        z = F.batch_norm(
            z,
            torch.zeros(self.bn2.running_mean.size()).to(self.dev),
            torch.ones(self.bn2.running_var.size()).to(self.dev),
            weights[4],
            weights[5],
            momentum=1,
            training=True,
        )

        y = x
        if self.skip:
            y = F.conv2d(
                input=y,
                weight=weights[6],
                bias=None,
                stride=self.stride,
                padding=self.padding,
            )

            y = F.batch_norm(
                y,
                torch.zeros(self.bn3.running_mean.size()).to(self.dev),
                torch.ones(self.bn3.running_var.size()).to(self.dev),
                weights[7],
                weights[8],
                momentum=1,
                training=True,
            )

        return F.relu(y + z)


class SegmentationClassifier(nn.Module):
    # one-layer minimal decoder used for segmentation benchmarks

    def __init__(self, in_channels, segm_classes, dev):
        super().__init__()
        self.in_channels = in_channels
        self.segm_classes = segm_classes
        self.dev = dev

        self.conv1 = nn.Conv2d(in_channels, self.segm_classes, kernel_size=1)

    def forward(self, x):
        z = self.conv1(x)
        return z

    def forward_weights(self, x, weights):
        z = F.conv2d(input=x, weight=weights[0], bias=weights[1])
        return z


class ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        dev,
        train_classes,
        eval_classes=None,
        segm_classes=2,
        num_keypoints=[5, 18, 16],
        criterion=nn.CrossEntropyLoss(),
        img_size=128,
        pretrained=False,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.train_classes = train_classes
        self.eval_classes = eval_classes
        self.segm_classes = segm_classes
        self.dev = dev
        self.criterion = criterion
        self.pretrained = pretrained

        if num_blocks == 10:
            layers = [1, 1, 1, 1]
            filters = [64, 128, 256, 512]
        elif num_blocks == 18:
            layers = [2, 2, 2, 2]
            filters = [64, 128, 256, 512]
        elif num_blocks == 34:
            layers = [3, 4, 6, 3]
            filters = [64, 128, 256, 512]
        else:
            raise ValueError(
                "ResNet layer not recognize. It must be resnet10, 18, or 34"
            )

        self.num_resunits = sum(layers)
        self.segm_feature_layers = list(accumulate(layers))

        self.conv = nn.Conv2d(
            in_channels=3,
            kernel_size=7,
            out_channels=64,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=64, momentum=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.num_keypoints = num_keypoints

        d = OrderedDict([])

        inpsize = img_size
        c = 0
        prev_filter = 64
        for idx, (layer, filter) in enumerate(zip(layers, filters)):
            stride = 1
            if idx == 0:
                in_channels = 64
            else:
                in_channels = filters[idx - 1]

            for i in range(layer):
                if i > 0:
                    in_channels = filter
                if stride == 2:
                    inpsize //= 2
                if prev_filter != filter:
                    stride = 2
                else:
                    stride = 1
                prev_filter = filter

                if inpsize % stride == 0:
                    padding = math.ceil(max((3 - stride), 0) / 2)
                else:
                    padding = math.ceil(max(3 - (inpsize % stride), 0) / 2)

                d.update(
                    {
                        f"res_block{c}": ResidualBlock(
                            in_channels=in_channels,
                            out_channels=filter,
                            stride=stride,
                            padding=padding,
                            dev=dev,
                        )
                    }
                )
                c += 1
        self.model = nn.ModuleDict({"features": nn.Sequential(d)})

        if self.pretrained:
            from torchvision.models import resnet18

            pre_trained_state_dict = resnet18(pretrained=True).state_dict()

            # match the keys
            own_model_keys = list(self.model.features.state_dict().keys())
            pretrained_model_keys_map = {}
            own_model_idx = 0
            for key in pre_trained_state_dict.keys():
                if "layer" not in key:
                    continue
                else:
                    own_model_current_key = own_model_keys[own_model_idx]
                    if key.split(".")[-1] == own_model_current_key.split(".")[-1]:
                        pretrained_model_keys_map[own_model_current_key] = key
                        own_model_idx += 1
                        if own_model_idx == len(own_model_keys):
                            break

            # now construct the modified state_dict
            remapped_state_dict = {}
            for key in pretrained_model_keys_map:
                remapped_state_dict[key] = pre_trained_state_dict[
                    pretrained_model_keys_map[key]
                ]

            assert len(remapped_state_dict) == len(remapped_state_dict)

            # manually fix several mismatches
            remapped_state_dict["res_block2.conv3.weight"] = remapped_state_dict[
                "res_block2.conv3.weight"
            ].repeat(1, 1, 3, 3)
            remapped_state_dict["res_block4.conv3.weight"] = remapped_state_dict[
                "res_block4.conv3.weight"
            ].repeat(1, 1, 3, 3)
            remapped_state_dict["res_block6.conv3.weight"] = remapped_state_dict[
                "res_block6.conv3.weight"
            ].repeat(1, 1, 3, 3)

            self.model.features.load_state_dict(remapped_state_dict)

        # Classifier for classification benchmarks
        rnd_input = torch.rand((1, 3, img_size, img_size))
        self.in_features = self.compute_in_features(rnd_input).size()[1]
        self.model.update(
            {
                "out": nn.Linear(
                    in_features=self.in_features, out_features=self.train_classes
                ).to(dev)
            }
        )

        # PPM module and classifier for segmentation benchmarks
        self.feat_dim = sum(filters)
        segm_out = SegmentationClassifier(self.feat_dim, self.segm_classes, dev)
        self.model.update({"segm_out": segm_out.to(dev)})

        # Regression layer for regression benchmarks
        self.model.update(
            {
                "regression_pose_animals_out": nn.Linear(
                    in_features=self.in_features, out_features=num_keypoints[0] * 2
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_pose_syn_out": nn.Linear(
                    in_features=self.in_features, out_features=num_keypoints[1] * 2
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_mpii_out": nn.Linear(
                    in_features=self.in_features, out_features=num_keypoints[2] * 2
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_distractor_out": nn.Linear(
                    in_features=self.in_features, out_features=2
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_pascal1d_out": nn.Linear(
                    in_features=self.in_features, out_features=1
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_shapenet1d_out": nn.Linear(
                    in_features=self.in_features, out_features=2
                ).to(dev)
            }
        )

        self.model.update(
            {
                "regression_shapenet2d_out": nn.Linear(
                    in_features=self.in_features, out_features=4
                ).to(dev)
            }
        )


    def compute_in_features(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = self.flatten(x)
        return x

    def forward(self, x, embedding=False, task_type=None):
        h, w = x.shape[-2], x.shape[-1]
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        inter_features, layer_idx = [], 0
        for i in range(self.num_resunits):
            x = self.model.features[i](x)
            if task_type == "segmentation":
                # take the feature maps after each group of residual blocks
                if (i + 1) == self.segm_feature_layers[layer_idx]:
                    inter_features.append(x)
                    layer_idx += 1
        if task_type == "segmentation":
            inter_features = [
                F.interpolate(
                    x, size=(h // 2, w // 2), mode="bilinear", align_corners=True
                )
                for x in inter_features
            ]
            x = torch.cat(inter_features, 1)
            if embedding:
                return x
            x = self.model.segm_out(x)
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        else:
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            x = self.flatten(x)
            if embedding:
                return x
            if task_type == "regression_shapenet1d":
                x = self.model.regression_shapenet1d_out(x)
                x = nn.Tanh()(x)
            elif task_type == "classification":
                x = self.model.out(x)
            elif task_type == "regression_distractor":
                x = self.model.regression_distractor_out(x)
                x = nn.Sigmoid()(x)
            elif task_type == "regression_pascal1d":
                x = self.model.regression_pascal1d_out(x)
                x = nn.Tanh()(x)
            elif task_type in "regression_pose_animals":
                x = self.model.regression_pose_animals_out(x)
            elif task_type in "regression_pose_animals_syn":
                x = self.model.regression_pose_syn_out(x)
            elif task_type in "regression_mpii":
                x = self.model.regression_mpii_out(x)
            elif task_type == "regression_shapenet2d":
                x = self.model.regression_shapenet2d_out(x)

        return x

    def forward_weights(
        self, x, weights, embedding=False, task_type=None, get_feature=False
    ):
        h, w = x.shape[-2], x.shape[-1]
        z = F.conv2d(input=x, weight=weights[0], bias=None, stride=2, padding=3)

        z = F.batch_norm(
            z,
            torch.zeros(self.bn.running_mean.size()).to(self.dev),
            torch.ones(self.bn.running_var.size()).to(self.dev),
            weights[1],
            weights[2],
            momentum=1,
            training=True,
        )

        z = F.relu(z)
        z = F.max_pool2d(z, kernel_size=3, stride=2, padding=1)

        lb = 3
        inter_features, layer_idx = [], 0
        for i in range(self.num_resunits):
            if self.model.features[i].skip:
                incr = 9
            else:
                incr = 6
            z = self.model.features[i].forward_weights(z, weights[lb : lb + incr])
            if task_type in ["segmentation"]:
                # take the feature maps after each group of residual blocks
                if (i + 1) == self.segm_feature_layers[layer_idx]:
                    inter_features.append(z)
                    layer_idx += 1
            lb += incr

        if task_type in ["segmentation"]:
            z = [
                F.interpolate(
                    x, size=(h // 2, w // 2), mode="bilinear", align_corners=True
                )
                for x in inter_features
            ]
            z = torch.cat(z, 1)
            if embedding:
                return z
            z = F.conv2d(z, weight=weights[-2], bias=weights[-1])
            z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=True)
        else:
            if get_feature:
                return z
            z = F.adaptive_avg_pool2d(z, output_size=(1, 1))
            z = self.flatten(z)
            if embedding:
                return z
            z = F.linear(z, weight=weights[-2], bias=weights[-1])
            if task_type in  ['regression_shapenet1d']:
                z = nn.Tanh()(z)
            elif task_type in  ['regression_distractor']:
                z = nn.Sigmoid()(z)
        return z

    def modify_out_layer(self, num_classes, segm_classes=None):
        # Reset the top layer weights
        if num_classes is None:
            num_classes = self.eval_classes
        if segm_classes is None:
            segm_classes = self.segm_classes
        self.model.out = nn.Linear(
            in_features=self.in_features, out_features=num_classes
        ).to(self.dev)
        self.model.out.bias = nn.Parameter(
            torch.zeros(self.model.out.bias.size(), device=self.dev)
        )

        # segmentation
        self.model.segm_out = nn.Conv2d(self.feat_dim, segm_classes, kernel_size=1).to(
            self.dev
        )
        self.model.segm_out.bias = nn.Parameter(
            torch.zeros(self.model.segm_out.bias.size(), device=self.dev)
        )

        # distractor
        self.model.regression_distractor_out = nn.Linear(
            in_features=self.in_features, out_features=2
        ).to(self.dev)
        self.model.regression_distractor_out.bias = nn.Parameter(
            torch.zeros(
                self.model.regression_distractor_out.bias.size(), device=self.dev
            )
        )

        # animal pose estimation
        self.model.regression_pose_animals_out = nn.Linear(
            in_features=self.in_features, out_features=self.num_keypoints[0] * 2
        ).to(self.dev)
        self.model.regression_pose_animals_out.bias = nn.Parameter(
            torch.zeros(
                self.model.regression_pose_animals_out.bias.size(), device=self.dev
            )
        )

        # synthetic animal pose estimation
        self.model.regression_pose_syn_out = nn.Linear(
            in_features=self.in_features, out_features=self.num_keypoints[1] * 2
        ).to(self.dev)
        self.model.regression_pose_syn_out.bias = nn.Parameter(
            torch.zeros(self.model.regression_pose_syn_out.bias.size(), device=self.dev)
        )

        # MPII
        self.model.regression_mpii_out = nn.Linear(
            in_features=self.in_features, out_features=self.num_keypoints[2] * 2
        ).to(self.dev)
        self.model.regression_mpii_out.bias = nn.Parameter(
            torch.zeros(self.model.regression_mpii_out.bias.size(), device=self.dev)
        )

        # pascal1d
        self.model.regression_pascal1d_out = nn.Linear(
            in_features=self.in_features, out_features=1
        ).to(self.dev)
        self.model.regression_pascal1d_out.bias = nn.Parameter(
            torch.zeros(self.model.regression_pascal1d_out.bias.size(), device=self.dev)
        )

        # shapenet1d
        self.model.regression_shapenet1d_out = nn.Linear(
            in_features=self.in_features, out_features=2
        ).to(self.dev)
        self.model.regression_shapenet1d_out.bias = nn.Parameter(
            torch.zeros(
                self.model.regression_shapenet1d_out.bias.size(), device=self.dev
            )
        )

        # shapenet2d
        self.model.regression_shapenet2d_out = nn.Linear(
            in_features=self.in_features, out_features=4
        ).to(self.dev)
        self.model.regression_shapenet2d_out.bias = nn.Parameter(
            torch.zeros(
                self.model.regression_shapenet2d_out.bias.size(), device=self.dev
            )
        )

    def freeze_layers(self, freeze, num_classes):
        if freeze:
            for name, param in self.named_parameters():
                # do not fine-tune the segmentation classifier
                # reset everything
                param.requires_grad = False
        self.modify_out_layer(num_classes)

    def load_params(self, state_dict):
        # delete weights of output head

        del state_dict["model.out.weight"]
        del state_dict["model.out.bias"]

        del state_dict["model.segm_out.conv1.weight"]
        del state_dict["model.segm_out.conv1.bias"]

        del state_dict["model.regression_shapenet1d_out.weight"]
        del state_dict["model.regression_shapenet1d_out.bias"]

        del state_dict["model.regression_distractor_out.weight"]
        del state_dict["model.regression_distractor_out.bias"]

        del state_dict["model.regression_pascal1d_out.weight"]
        del state_dict["model.regression_pascal1d_out.bias"]

        del state_dict["model.regression_shapenet2d_out.weight"]
        del state_dict["model.regression_shapenet2d_out.bias"]

        del state_dict["model.regression_pose_animals_out.weight"]
        del state_dict["model.regression_pose_animals_out.bias"]

        del state_dict["model.regression_pose_syn_out.weight"]
        del state_dict["model.regression_pose_syn_out.bias"]

        del state_dict["model.regression_mpii_out.weight"]
        del state_dict["model.regression_mpii_out.bias"]

        self.load_state_dict(state_dict, strict=False)
