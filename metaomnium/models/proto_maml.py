import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import inverse as inv
from torch import mm
from torch import transpose as t
from torch.autograd import Variable

from .learner import Learner
from .modules.utils import (
    accuracy,
    get_loss_and_grads,
    miou,
    put_on_device,
    regression_loss,
)


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


class ProtoMAML(Learner):
    def __init__(
        self, base_lr, second_order=False, grad_clip=None, meta_batch_size=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.grad_clip = grad_clip
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size

        self.task_counter = 0
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [
            p.clone().detach().to(self.dev) for p in self.baselearner.parameters()
        ]
        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(self.dev)

        self.grad_buffer = [
            torch.zeros(p.size(), device=self.dev) for p in self.initialization
        ]

        for p in self.initialization:
            p.requires_grad = True

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.initialization, lr=self.lr, momentum=self.momentum
            )
        else:
            self.optimizer = torch.optim.Adam(self.initialization, lr=self.lr)
        self.output_dim = self.baselearner.model.out.in_features
        self.lambda_rr = LambdaLayer(
            kwargs["learn_lambda"], kwargs["init_lambda"], kwargs["lambda_base"]
        )
        self.n_augment = 1

    def _fast_weights(self, params, gradients):
        if self.grad_clip is not None:
            gradients = [
                torch.clamp(p, -self.grad_clip, +self.grad_clip) for p in gradients
            ]

        fast_weights = [
            params[i] - self.base_lr * gradients[i] for i in range(len(gradients))
        ]

        return fast_weights

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

    def _deploy(
        self,
        learner,
        train_x,
        train_y,
        test_x,
        test_y,
        T,
        fast_weights,
        task_type,
        num_classes=None,
        train_mode=False,
    ):
        loss_history = list()

        # initialize the classifier weights with class prototypes
        if train_mode:
            num_classes = self.baselearner.train_classes
        else:
            if num_classes is None:
                num_classes = self.baselearner.eval_classes

        learner.zero_grad()
        support_embeddings = learner.forward_weights(
            train_x, fast_weights, embedding=True, task_type=task_type
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
                    (num_classes, support_embeddings.size(1)), device=learner.dev
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
            fast_weights.append(output_weight.unsqueeze(dim=2).unsqueeze(dim=3))
        else:
            fast_weights.append(output_weight)
        fast_weights.append(output_bias)

        for _ in range(T):
            xinp, yinp = train_x, train_y

            loss, grads = get_loss_and_grads(
                learner,
                xinp,
                yinp,
                weights=fast_weights,
                create_graph=self.second_order,
                retain_graph=T > 1 or self.second_order,
                flat=False,
                task_type=task_type,
            )

            loss_history.append(loss)
            fast_weights = self._fast_weights(params=fast_weights, gradients=grads)

        xinp, yinp = test_x, test_y

        out = learner.forward_weights(xinp, fast_weights, task_type=task_type)
        if task_type.startswith("regression"):
            if task_type == "regression_pascal1d":
                out = out[..., :2]
                yinp = yinp[..., :2]
            test_loss = regression_loss(out, yinp, task_type, mode="train")
        else:
            test_loss = learner.criterion(out, yinp)
        loss_history.append(test_loss.item())
        with torch.no_grad():
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            if task_type == "segmentation":
                score = miou(preds, test_y)
            elif task_type.startswith("regression"):
                if task_type == "regression_pascal1d":
                    out = out[..., :2]
                    test_y = test_y[..., :2]
                score = regression_loss(out, test_y, task_type, mode="eval")
            else:
                score = accuracy(preds, test_y)

        return score, test_loss, loss_history, probs.cpu().numpy(), preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type):
        self.baselearner.train()
        self.task_counter += 1

        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        fast_weights = [p.clone() for p in self.initialization]
        # Get the weights only used for the given task
        filtered_fast_weights = fast_weights[:-18]
        score, test_loss, _, probs, preds = self._deploy(
            self.baselearner,
            train_x,
            train_y,
            test_x,
            test_y,
            self.T,
            filtered_fast_weights,
            task_type,
            train_mode=True,
        )

        test_loss.backward()

        if self.grad_clip is not None:
            for p in self.initialization:
                if p.grad is not None:
                    p.grad = torch.clamp(p.grad, -self.grad_clip, +self.grad_clip)
                else:
                    p.grad = torch.zeros_like(p)

        self.grad_buffer = [
            self.grad_buffer[i] + self.initialization[i].grad
            for i in range(len(self.initialization))
        ]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0:
            for i, p in enumerate(self.initialization):
                p.grad = self.grad_buffer[i]
            self.optimizer.step()

            self.grad_buffer = [
                torch.zeros(p.size(), device=self.dev) for p in self.initialization
            ]
            self.task_counter = 0
            self.optimizer.zero_grad()

        return score, test_loss.item(), probs, preds

    def evaluate(
        self, num_classes, train_x, train_y, test_x, test_y, val=True, task_type=None
    ):
        if num_classes is None:
            self.baselearner.eval()
            fast_weights = [p.clone() for p in self.initialization]
            learner = self.baselearner
        else:
            self.val_learner.load_params(self.baselearner.state_dict())
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_classes)

            fast_weights = [p.clone() for p in self.initialization]
            learner = self.val_learner

        # Get the weights only used for the given task
        filtered_fast_weights = fast_weights[:-18]

        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )
        if val:
            T = self.T_val
        else:
            T = self.T_test

        score, _, loss_history, probs, preds = self._deploy(
            learner,
            train_x,
            train_y,
            test_x,
            test_y,
            T,
            filtered_fast_weights,
            task_type=task_type,
            num_classes=num_classes,
            train_mode=False,
        )

        return score, loss_history, probs, preds

    def dump_state(self):
        return [p.clone().detach() for p in self.initialization]

    def load_state(self, state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
