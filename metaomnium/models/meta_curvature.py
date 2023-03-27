import numpy as np
import torch
import torch.nn.functional as F

from .learner import Learner
from .modules.utils import accuracy, get_loss_and_grads, miou, put_on_device, regression_loss


class MetaCurvatureTransform(torch.nn.Module):
    """
    Implements the Meta-Curvature transform of Park and Oliva, 2019.
    
    Taken from https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/module_transform.py
    """

    def __init__(self, param, lr=1.0):
        super(MetaCurvatureTransform, self).__init__()
        self.lr = lr
        shape = param.shape
        if len(shape) == 1:  # bias
            self.dim = 1
            self.mc = torch.nn.Parameter(torch.ones_like(param))
        elif len(shape) == 2:  # FC
            self.dim = 2
            self.mc_in = torch.nn.Parameter(torch.eye(shape[0]))
            self.mc_out = torch.nn.Parameter(torch.eye(shape[1]))
        elif len(shape) == 4:  # CNN
            self.dim = 4
            self.n_in = shape[0]
            self.n_out = shape[1]
            self.n_f = int(np.prod(shape) / (self.n_in * self.n_out))
            self.mc_in = torch.nn.Parameter(torch.eye(self.n_in))
            self.mc_out = torch.nn.Parameter(torch.eye(self.n_out))
            self.mc_f = torch.nn.Parameter(torch.eye(self.n_f))
        else:
            raise NotImplementedError('Parameter with shape',
                                      shape,
                                      'is not supported by MetaCurvature.')

    def forward(self, grad):
        if self.dim == 1:
            update = self.mc * grad
        elif self.dim == 2:
            update = self.mc_in @ grad @ self.mc_out
        else:
            update = grad.permute(2, 3, 0, 1).contiguous()
            shape = update.shape
            update = update.view(-1, self.n_out) @ self.mc_out
            update = self.mc_f @ update.view(self.n_f, -1)
            update = update.view(self.n_f, self.n_in, self.n_out)
            update = update.permute(1, 0, 2).contiguous().view(self.n_in, -1)
            update = self.mc_in @ update
            update = update.view(
                self.n_in,
                self.n_f,
                self.n_out).permute(1, 0, 2).contiguous().view(shape)
            update = update.permute(2, 3, 0, 1).contiguous()
        return self.lr * update


class MetaCurvature(Learner):

    def __init__(self, base_lr, second_order=False, grad_clip=None,
                 meta_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.grad_clip = grad_clip
        self.second_order = second_order
        self.meta_batch_size = meta_batch_size

        self.task_counter = 0
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)
        self.initialization = [p.clone().detach().to(self.dev) for p in
                               self.baselearner.parameters()]

        transforms_modules = []
        for p in self.initialization:
            transforms_modules.append(MetaCurvatureTransform(p))
        self.transforms_modules = torch.nn.ModuleList(
            transforms_modules).to(self.dev)

        self.val_learner = self.baselearner_fn(**self.baselearner_args).to(
            self.dev)

        opt_params = list(self.initialization) + list(self.transforms_modules.parameters())
        self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in opt_params]

        for p in opt_params:
            p.requires_grad = True

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                opt_params, lr=self.lr, momentum=self.momentum)
        else:
            self.optimizer = torch.optim.Adam(
                opt_params, lr=self.lr)

    def _fast_weights(self, params, gradients):
        if self.grad_clip is not None:
            gradients = [torch.clamp(p, -self.grad_clip, +self.grad_clip) for p
                         in gradients]

        fast_weights = [params[i] - self.base_lr * gradients[i] for i in
                        range(len(gradients))]

        return fast_weights

    def _deploy(self, learner, train_x, train_y, test_x, test_y, T, fast_weights, task_type):
        loss_history = list()

        for _ in range(T):
            xinp, yinp = train_x, train_y

            loss, grads = get_loss_and_grads(learner, xinp, yinp,
                                             weights=fast_weights, create_graph=self.second_order,
                                             retain_graph=T > 1 or self.second_order, flat=False, task_type=task_type)

            # Apply MetaCurvature transform on grads
            if task_type == "classification":
                selected_modules = self.transforms_modules[:-16]
            elif task_type == "segmentation":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-16:-14])
            elif task_type == "regression_pose_animals":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-14:-12])
            elif task_type == 'regression_pose_animals_syn':
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-12:-10])
            elif task_type == "regression_mpii":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-10:-8])
            elif task_type == "regression_distractor":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-8:-6])
            elif task_type == "regression_pascal1d":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-6:-4])
            elif task_type == "regression_shapenet1d":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-4:-2])
            elif task_type == "regression_shapenet2d":
                selected_modules = self.transforms_modules[:-18]
                selected_modules.extend(self.transforms_modules[-2:])

            transformed_grads = []
            for idx, grad in enumerate(grads):
                transformed_grads.append(selected_modules[idx](grad))

            loss_history.append(loss)
            fast_weights = self._fast_weights(
                params=fast_weights, gradients=transformed_grads)

        xinp, yinp = test_x, test_y

        out = learner.forward_weights(xinp, fast_weights, task_type=task_type)

        if task_type.startswith("regression"):
            test_loss = regression_loss(out, yinp, task_type, mode="train")
        else:
            test_loss = learner.criterion(out, yinp)

        loss_history.append(test_loss.item())
        with torch.no_grad():
            probs = F.softmax(out, dim=1)
            preds = torch.argmax(probs, dim=1)
            if task_type.startswith("regression"):
                score = regression_loss(out, yinp, task_type, mode="eval")
            elif task_type == "segmentation":
                score = miou(preds, test_y)
            else:
                score = accuracy(preds, test_y)

        return score, test_loss, loss_history, probs.cpu().numpy(), \
               preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type):
        self.baselearner.train()
        self.task_counter += 1

        train_x, train_y, test_x, test_y = put_on_device(self.dev, [train_x,
                                                                    train_y, test_x, test_y])

        fast_weights = [p.clone() for p in self.initialization]

        # Get the weights only used for the given task
        if task_type == "classification":
            filtered_fast_weights = fast_weights[:-16]
        elif task_type == "segmentation":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-16:-14]
        elif task_type == "regression_pose_animals":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-14:-12]
        elif task_type == 'regression_pose_animals_syn':
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-12:-10]
        elif task_type == "regression_mpii":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-10:-8]
        elif task_type == "regression_distractor":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-8:-6]
        elif task_type == "regression_pascal1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-6:-4]
        elif task_type == "regression_shapenet1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-4:-2]
        elif task_type == "regression_shapenet2d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-2:]

        score, test_loss, _, probs, preds = self._deploy(self.baselearner,
                                                         train_x, train_y, test_x, test_y, self.T,
                                                         filtered_fast_weights, task_type)

        test_loss.backward()

        opt_params = list(self.initialization) + list(self.transforms_modules.parameters())
        if self.grad_clip is not None:
            for p in opt_params:
                if p.grad is not None:
                    p.grad = torch.clamp(
                        p.grad, -self.grad_clip, +self.grad_clip)
                else:
                    p.grad = torch.zeros_like(p)

        self.grad_buffer = [self.grad_buffer[i] + opt_params[i].grad
                            for i in range(len(opt_params))]
        self.optimizer.zero_grad()

        if self.task_counter % self.meta_batch_size == 0:
            for i, p in enumerate(opt_params):
                p.grad = self.grad_buffer[i]
            self.optimizer.step()

            self.grad_buffer = [torch.zeros(p.size(), device=self.dev) for p in
                                opt_params]
            self.task_counter = 0
            self.optimizer.zero_grad()

        return score, test_loss.item(), probs, preds

    def evaluate(self, num_classes, train_x, train_y, test_x, test_y,
                 val=True, task_type=None):
        if num_classes is None:
            self.baselearner.eval()
            fast_weights = [p.clone() for p in self.initialization]
            learner = self.baselearner
        else:
            self.val_learner.load_params(self.baselearner.state_dict())
            self.val_learner.eval()
            self.val_learner.modify_out_layer(num_classes)

            fast_weights = [p.clone() for p in self.initialization[:-18]]
            initialization = [p.clone().detach().to(self.dev) for p in
                              self.val_learner.parameters()]
            fast_weights.extend(initialization[-18:])
            for p in fast_weights[-18:]:
                p.requires_grad = True
            learner = self.val_learner

        # Get the weights only used for the given task
        if task_type == "classification":
            filtered_fast_weights = fast_weights[:-16]
        elif task_type == "segmentation":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-16:-14]
        elif task_type == "regression_pose_animals":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-14:-12]
        elif task_type == 'regression_pose_animals_syn':
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-12:-10]
        elif task_type == "regression_mpii":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-10:-8]
        elif task_type == "regression_distractor":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-8:-6]
        elif task_type == "regression_pascal1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-6:-4]
        elif task_type == "regression_shapenet1d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-4:-2]
        elif task_type == "regression_shapenet2d":
            filtered_fast_weights = fast_weights[:-18] + fast_weights[-2:]

        train_x, train_y, test_x, test_y = put_on_device(self.dev,
                                                         [train_x, train_y, test_x, test_y])
        if val:
            T = self.T_val
        else:
            T = self.T_test

        score, _, loss_history, probs, preds = self._deploy(learner, train_x,
                                                            train_y, test_x, test_y, T, filtered_fast_weights,
                                                            task_type=task_type)

        return score, loss_history, probs, preds

    def dump_state(self):
        return [[p.clone().detach() for p in self.initialization], self.transforms_modules.state_dict()]

    def load_state(self, state):
        self.initialization = [p.clone() for p in state[0]]
        self.transforms_modules.load_state_dict(state[1])
        for p in self.initialization:
            p.requires_grad = True
        for p in self.transforms_modules.parameters():
            p.requires_grad = True
