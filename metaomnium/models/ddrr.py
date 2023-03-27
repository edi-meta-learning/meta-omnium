import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import transpose as t
from torch import inverse as inv
from torch import mm
import numpy as np
from torch.autograd import Variable
from .learner import Learner
from .modules.utils import (
    accuracy,
    miou,
    put_on_device,
    empty_context,
    regression_loss,
)


# Ridge Regression from the paper: Meta-learning with differentiable closed-form solvers
# Source code: https://github.com/bertinetto/r2d2


def t_(x):
    return t(x, 0, 1)


class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=False, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]).cuda()
        self.base = base
        if learn_lambda:
            self.l = nn.Parameter(self.l)
        else:
            self.l = Variable(self.l)

    def forward(self, x):
        if self.base == 1:
            return x * self.l
        else:
            return x * (self.base**self.l)


class AdjustLayer(nn.Module):
    def __init__(self, init_scale=1e-4, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale]).cuda())
        self.bias = nn.Parameter(torch.FloatTensor([init_bias]).cuda())
        self.base = base

    def forward(self, x):
        if self.base == 1:
            return x * self.scale + self.bias
        else:
            return x * (self.base**self.scale) + self.base**self.bias - 1


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples, n_way).zero_()
    for i in range(n_way):
        label[i::n_way, i] = 1
    return to_variable(label)


def make_long_label(n_way, n_samples):
    label = torch.LongTensor(n_way * n_samples).zero_()
    for i in range(n_way * n_samples):
        label[i] = i // n_samples
    return to_variable(label)


def shuffle_queries_multi(x, n_way, n_shot, n_query, n_augment, y_binary, y):
    ind_xs = torch.linspace(
        0, n_way * n_shot * n_augment - 1, steps=n_way * n_shot * n_augment
    ).long()
    ind_xs = Variable(ind_xs.cuda())
    perm_xq = torch.randperm(n_way * n_query).long()
    perm_xq = Variable(perm_xq.cuda())
    permute = torch.cat([ind_xs, perm_xq + len(ind_xs)])
    x = x[permute, :, :, :]
    y_binary = y_binary[perm_xq, :]
    y = y[perm_xq]
    return x, y_binary, y


class DDRR(Learner):
    def __init__(self, meta_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.meta_batch_size = meta_batch_size
        self.task_counter = 0
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [
            p.clone().detach().to(self.dev) for p in self.baselearner.parameters()
        ]

        for p in self.initialization:
            p.requires_grad = True

        self.adjust = AdjustLayer(
            init_scale=kwargs["init_adj_scale"], base=kwargs["adj_base"]
        )
        self.lambda_rr = LambdaLayer(
            kwargs["learn_lambda"], kwargs["init_lambda"], kwargs["lambda_base"]
        )

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.initialization
                + list(self.adjust.parameters())
                + list(self.lambda_rr.parameters()),
                lr=self.lr,
                momentum=self.momentum,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.initialization
                + list(self.adjust.parameters())
                + list(self.lambda_rr.parameters()),
                lr=self.lr,
            )

        self.n_way = kwargs["baselearner_args"]["train_classes"]
        self.n_augment = 1

        self.output_dim = self.baselearner.model.out.in_features

        # Using BCE Loss for segmentation, Cannot use CE because RR solution predicts the segmentation mask directly
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def _deploy(
        self,
        train_x,
        train_y,
        test_x,
        test_y,
        train_mode,
        num_classes=None,
        task_type=None,
    ):
        if train_mode:
            num_classes = self.baselearner.train_classes
        else:
            if num_classes is None:
                num_classes = self.baselearner.eval_classes

        if task_type == "classification":
            # Get number of shots and query samples, Number of shots can be give by train_x.shape[0]/n_way
            n_way = self.n_way
            n_shot, n_query = int(train_x.size(0) / n_way), int(test_x.size(0) / n_way)
            # one hot Label for "inner loop" closed form solution -> scaled version
            y_inner = make_float_label(self.n_way, n_shot * self.n_augment) / np.sqrt(
                self.n_way * n_shot * self.n_augment
            )
            # one hot Label for query set (outer loop)
            y_outer_binary = make_float_label(self.n_way, n_query)

        elif task_type == "segmentation":
            n_way = 1
            n_shot, n_query = int(train_x.size(0) / n_way), int(test_x.size(0) / n_way)
            num_pixels = train_x.size(2) * train_x.size(3)
            n_shot = n_shot * num_pixels

        elif task_type.startswith("regression"):
            n_way = 1
            n_shot, n_query = int(train_x.size(0) / n_way), int(test_x.size(0) / n_way)
            if task_type == "regression_shapenet1d":
                y_inner = train_y[..., :2]
            else:
                y_inner = train_y
        # Get support and query features using the forward weight function with self.initialization
        zs = self.baselearner.forward_weights(
            train_x, self.initialization, embedding=True, task_type=task_type
        )
        zq = self.baselearner.forward_weights(
            test_x, self.initialization, embedding=True, task_type=task_type
        )

        if task_type == "segmentation":
            rr_output_dim = zs.shape[1]
        else:
            if task_type == "classification":
                # Original paper does this for classification
                zs /= np.sqrt(n_way * n_shot * self.n_augment)
            rr_output_dim = self.output_dim

        # Select standard or woodbury solution
        if n_way * n_shot * self.n_augment > rr_output_dim + 1:
            rr_type = "standard"
            I = Variable(torch.eye(rr_output_dim + 1).cuda())
        else:
            rr_type = "woodbury"
            I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())

        if task_type == "segmentation":
            # add a column of ones for the bias
            train_y = train_y.unsqueeze(1).float()
            test_y = test_y.unsqueeze(1).float()
            train_y = F.interpolate(
                train_y, size=(112, 112), mode="bilinear", align_corners=True
            )
            test_y = F.interpolate(
                test_y, size=(112, 112), mode="bilinear", align_corners=True
            )

            block_size = 28
            out = put_on_device(self.dev, [torch.zeros(test_y.shape)])[0]

            for hr in range(0, zs.shape[-2], block_size):
                for vr in range(0, zs.shape[-1], block_size):
                    zs_block = zs[:, :, hr : hr + block_size, vr : vr + block_size]
                    zq_block = zq[:, :, hr : hr + block_size, vr : vr + block_size]
                    y_inner_block = train_y[
                        :, :, hr : hr + block_size, vr : vr + block_size
                    ]
                    y_outer_block = test_y[
                        :, :, hr : hr + block_size, vr : vr + block_size
                    ]
                    zs_block, zq_block, y_inner_block, y_outer_block = process_for_seg(
                        zs_block, zq_block, y_inner_block, y_outer_block
                    )
                    ones = Variable(
                        torch.unsqueeze(torch.ones(zs_block.size(0)).cuda(), 1)
                    )
                    if rr_type == "woodbury":
                        wb = self.rr_woodbury(
                            torch.cat((zs_block, ones), 1),
                            n_way,
                            n_shot,
                            I,
                            y_inner_block,
                        )

                    else:
                        wb = self.rr_standard(
                            torch.cat((zs_block, ones), 1),
                            n_way,
                            n_shot,
                            I,
                            y_inner_block,
                        )

                    # Split to get weight and bias
                    w = wb.narrow(dim=0, start=0, length=rr_output_dim)
                    b = wb.narrow(dim=0, start=rr_output_dim, length=1)
                    # Regress to get query set labels
                    out_block = mm(zq_block, w) + b
                    out_block = out_block.reshape(
                        test_x.shape[0], 1, block_size, block_size
                    )
                    out[:, :, hr : hr + block_size, vr : vr + block_size] = out_block

        else:
            ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
            if rr_type == "woodbury":
                wb = self.rr_woodbury(
                    torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner
                )

            else:
                wb = self.rr_standard(
                    torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner
                )

            # Split to get weight and bias
            w = wb.narrow(dim=0, start=0, length=rr_output_dim)
            b = wb.narrow(dim=0, start=rr_output_dim, length=1)
            # Regress to get query set labels
            out = mm(zq, w) + b

        if task_type.startswith("regression"):
            if task_type in ["regression_shapenet1d"]:
                out = nn.Tanh()(out)
            elif task_type in ['regression_distractor']:
                out = nn.Sigmoid()(out)

        # Better to adjust features for segmentations for a smaller loss
        if task_type == "segmentation":
            out = self.adjust(out)

        # Loss for query set labels
        if task_type == "classification":
            loss = self.baselearner.criterion(out, test_y)
        elif task_type == "segmentation":
            loss = self.seg_loss(out, test_y)
        elif task_type.startswith("regression"):
            loss = regression_loss(out, test_y, task_type, mode="train")

        with torch.no_grad():
            if task_type == "classification":
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                score = accuracy(preds, test_y)
            elif task_type == "segmentation":
                out = out.squeeze(1)
                test_y = test_y.squeeze(1)
                probs = self.sigmoid(out)
                preds = put_on_device(self.dev, [torch.zeros(test_y.shape)])[0]
                preds[probs > torch.mean(probs)] = 1.0
                score = miou(preds, test_y)

            elif task_type.startswith("regression"):
                score = regression_loss(out, test_y, task_type, mode="eval")

            if task_type.startswith("regression"):
                probs = torch.zeros(out.shape).cuda()
                preds = out.detach()

        return score, loss, probs.cpu().numpy(), preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type):
        self.baselearner.train()
        self.task_counter += 1
        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        acc, loss, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, train_mode=True, task_type=task_type
        )
        loss.backward()
        if self.task_counter % self.meta_batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return acc, loss.item(), probs, preds

    def evaluate(
        self, num_classes, train_x, train_y, test_x, test_y, val=True, task_type=None
    ):
        # Num classes is n-way
        self.baselearner.eval()
        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        acc, loss, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, train_mode=False, task_type=task_type
        )

        acc = np.float64(acc)
        loss = np.float64(loss.item())
        return acc, [loss], probs, preds

    def dump_state(self):
        return [
            p.clone().detach()
            for p in self.initialization + list(self.adjust.parameters())
        ]

    def load_state(self, state):
        self.initialization = [p.clone() for p in state[:-2]]
        for p in self.initialization:
            p.requires_grad = True
        self.adjust.scale.data = state[-2].data
        self.adjust.bias.data = state[-1].data

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys=False):
        if not linsys:
            w = mm(
                mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary
            )
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = gesv(v, A)

        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys=False):
        if not linsys:
            w = mm(
                mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary
            )
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = gesv(v, A)
            w = mm(t_(x), w_)

        return w


def process_for_seg(zs, zq, train_y, test_y):
    zs = torch.flatten(zs, start_dim=2, end_dim=3)
    zq = torch.flatten(zq, start_dim=2, end_dim=3)

    h, w = train_y.shape[-2], train_y.shape[-1]
    y_inner = torch.flatten(train_y, start_dim=2, end_dim=3)

    zs = zs.permute(0, 2, 1)
    zs = zs.reshape(-1, zs.shape[-1])

    zq = zq.permute(0, 2, 1)
    zq = zq.reshape(-1, zq.shape[-1])

    y_inner = y_inner.permute(0, 2, 1)
    y_inner = y_inner.reshape(-1, y_inner.shape[-1])

    y_outer = torch.flatten(test_y, start_dim=2, end_dim=3)
    y_outer = y_outer.permute(0, 2, 1)
    y_outer = y_outer.reshape(-1, y_outer.shape[-1])

    return zs, zq, y_inner, y_outer
