import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse as inv
from torch import mm
from torch import transpose as t
from torch.autograd import Variable

from .learner import Learner
from .modules.utils import deploy_on_task, put_on_device, update


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


class ProtoFineTuning(Learner):
    def __init__(
        self, freeze, test_lr=0.001, test_opt="adam", beta1=0.9, beta2=0.999, **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze = freeze
        self.test_opt = test_opt
        self.test_lr = test_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.baselearner.train()
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.val_learner.load_state_dict(self.baselearner.state_dict())
        self.val_learner.train()

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.baselearner.parameters(), lr=self.lr, momentum=self.momentum
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.baselearner.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
            )
        self.output_dim = self.baselearner.model.out.in_features
        self.lambda_rr = LambdaLayer(
            kwargs["learn_lambda"], kwargs["init_lambda"], kwargs["lambda_base"]
        )
        self.n_augment = 1

    def _get_features(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:], mode="bilinear")
        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (
            mask[None, ...].sum(dim=(2, 3)) + 1e-5
        )  # 1 x C
        return masked_fts

    def _get_prototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype

    def _rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys=False):
        if not linsys:
            w = mm(
                mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary
            )
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = gesv(v, A)

        return w

    def _rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys=False):
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

    def train(self, train_x, train_y, task_type=None, **kwargs):
        self.baselearner.train()
        train_x, train_y = put_on_device(self.dev, [train_x, train_y])
        acc, loss, probs, preds = update(
            self.baselearner, self.optimizer, train_x, train_y, task_type=task_type
        )
        return acc, loss, probs, preds

    def evaluate(
        self, num_classes, train_x, train_y, test_x, test_y, val=True, task_type=None
    ):
        self.val_learner.load_params(self.baselearner.state_dict())
        self.val_learner.eval()
        self.val_learner.freeze_layers(self.freeze, num_classes)

        # Put on the right device
        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        support_embeddings = self.val_learner(
            train_x, embedding=True, task_type=task_type
        )

        if task_type.startswith("regression"):
            n_way = 1
            n_shot = train_x.shape[0]
            rr_type = "woodbury"
            if task_type == "regression_shapenet1d":
                y_inner = train_y[..., :2]
            else:
                y_inner = train_y
            rr_output_dim = self.output_dim
            I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())
            ones = Variable(
                torch.unsqueeze(torch.ones(support_embeddings.size(0)).cuda(), 1)
            )
            if rr_type == "woodbury":
                wb = self._rr_woodbury(
                    torch.cat((support_embeddings, ones), 1), n_way, n_shot, I, y_inner
                )

            else:
                wb = self._rr_standard(
                    torch.cat((support_embeddings, ones), 1), n_way, n_shot, I, y_inner
                )
            # Split to get weight and bias
            w = wb.narrow(dim=0, start=0, length=rr_output_dim)
            b = wb.narrow(dim=0, start=rr_output_dim, length=1)
            output_weight = w.t().detach().requires_grad_()
            output_bias = b.squeeze().detach().requires_grad_()
        else:
            if task_type == "segmentation":
                # Extract right features
                supp_fg_fts = [
                    [
                        self._get_features(
                            support_embeddings[e].unsqueeze(dim=0),
                            train_y[e].unsqueeze(dim=0),
                        )
                        for e in range(train_y.shape[0])
                    ]
                ]
                supp_bg_fts = [
                    [
                        self._get_features(
                            support_embeddings[e].unsqueeze(dim=0),
                            1 - train_y[e].unsqueeze(dim=0),
                        )
                        for e in range(train_y.shape[0])
                    ]
                ]

                # Obtain prototypes
                fg_prototypes, bg_prototype = self._get_prototype(
                    supp_fg_fts, supp_bg_fts
                )

                # Compute the distances
                prototypes = torch.cat((bg_prototype, fg_prototypes[0]))

            else:
                prototypes = torch.zeros(
                    (num_classes, support_embeddings.size(1)), device=self.dev
                )
                for class_id in range(num_classes):
                    mask = train_y == class_id
                    prototypes[class_id] = (
                        support_embeddings[mask].sum(dim=0) / torch.sum(mask).item()
                    )

            # Create output layer weights with prototype-based initialization
            init_weight = 2 * prototypes
            init_bias = -torch.norm(prototypes, dim=1) ** 2
            output_weight = init_weight.detach().requires_grad_()
            output_bias = init_bias.detach().requires_grad_()

        if task_type == "segmentation":
            self.val_learner.model.segm_out.weight.data = (
                output_weight.unsqueeze(dim=2).unsqueeze(dim=3).data
            )
            self.val_learner.model.segm_out.bias.data = output_bias.data
        elif task_type.startswith("regression"):
            if task_type == "regression_shapenet1d":
                self.val_learner.model.regression_shapenet1d_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_shapenet1d_out.bias.data = (
                    output_bias.data
                )
            elif task_type == "regression_distractor":
                self.val_learner.model.regression_distractor_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_distractor_out.bias.data = (
                    output_bias.data
                )
            elif task_type == "regression_pascal1d":
                self.val_learner.model.regression_pascal1d_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_pascal1d_out.bias.data = (
                    output_bias.data
                )
            elif task_type == "regression_shapenet2d":
                self.val_learner.model.regression_shapenet2d_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_shapenet2d_out.bias.data = (
                    output_bias.data
                )
            elif task_type in "regression_pose_animals":
                self.val_learner.model.regression_pose_animals_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_pose_animals_out.bias.data = (
                    output_bias.data
                )
            elif task_type in "regression_pose_animals_syn":
                self.val_learner.model.regression_pose_syn_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_pose_syn_out.bias.data = (
                    output_bias.data
                )
            elif task_type in "regression_mpii":
                self.val_learner.model.regression_mpii_out.weight.data = (
                    output_weight.data
                )
                self.val_learner.model.regression_mpii_out.bias.data = output_bias.data
        else:
            self.val_learner.model.out.weight.data = output_weight.data
            self.val_learner.model.out.bias.data = output_bias.data

        if self.test_opt == "sgd":
            val_optimizer = torch.optim.SGD(
                self.val_learner.parameters(), lr=self.test_lr, momentum=self.momentum
            )
        else:
            val_optimizer = torch.optim.Adam(
                self.val_learner.parameters(),
                lr=self.test_lr,
                betas=(self.beta1, self.beta2),
            )

        if val:
            T = self.T_val
        else:
            T = self.T_test

        # Train on support set and get loss on query set
        test_acc, loss_history, probs, preds = deploy_on_task(
            model=self.val_learner,
            optimizer=val_optimizer,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            T=T,
            test_batch_size=self.test_batch_size,
            task_type=task_type,
        )

        return test_acc, loss_history, probs, preds

    def dump_state(self):
        return {k: v.clone() for k, v in self.baselearner.state_dict().items()}

    def load_state(self, state):
        self.baselearner.eval()
        self.baselearner.load_state_dict(state)
