import contextlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.spatial.transform import Rotation as R


def accuracy(y_pred, y):
    return ((y_pred == y).float().sum() / len(y)).item()


def miou(y_pred, y):
    num_classes = 2
    assert y_pred.dim() in [1, 2, 3]
    assert y_pred.shape == y.shape
    output = y_pred.view(-1)
    target = y.view(-1)
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection, bins=num_classes, min=0, max=num_classes - 1
    )
    area_output = torch.histc(output, bins=num_classes, min=0, max=num_classes - 1)
    area_target = torch.histc(target, bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_output + area_target - area_intersection + 1e-10

    area_intersection, area_union = (
        area_intersection.cpu().numpy(),
        area_union.cpu().numpy(),
    )

    iou_scores = area_intersection / area_union

    foreground_iou = iou_scores[1]

    return foreground_iou


def dist_acc(dists, thr=0.01):
    """Return percentage below threshold while ignoring values with a -1"""
    dists = dists.detach().cpu().numpy()
    dist_cal = np.not_equal(dists, 0.0)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def azimuth_loss(q_gt, q_pr):
    loss = torch.mean(torch.sum((q_gt[..., :2] - q_pr) ** 2, dim=-1))
    return loss


def get_batch(train_x, train_y, batch_size):
    if batch_size is None:
        return train_x, train_y
    batch_indices = np.random.randint(0, train_x.size()[0], batch_size)
    x_batch, y_batch = train_x[batch_indices], train_y[batch_indices]
    return x_batch, y_batch


def degree_loss(q_gt, q_pr):
    q_gt = torch.rad2deg(q_gt[..., -1])
    pr_cos = q_pr[..., 0]
    pr_sin = q_pr[..., 1]
    ps_sin = torch.where(pr_sin >= 0)
    ng_sin = torch.where(pr_sin < 0)
    pr_deg = torch.acos(pr_cos)
    pr_deg_ng = -torch.acos(pr_cos) + 2 * math.pi
    pr_deg[ng_sin] = pr_deg_ng[ng_sin]
    pr_deg = torch.rad2deg(pr_deg)
    errors = torch.stack(
        (
            torch.abs(q_gt - pr_deg),
            torch.abs(q_gt + 360.0 - pr_deg),
            torch.abs(q_gt - (pr_deg + 360.0)),
        ),
        dim=-1,
    )
    errors, _ = torch.min(errors, dim=-1)
    losses = torch.mean(errors)
    return losses


def degree_loss(q_gt, q_pr):
    q_gt = torch.rad2deg(q_gt[..., -1])
    pr_cos = q_pr[..., 0]
    pr_sin = q_pr[..., 1]
    ps_sin = torch.where(pr_sin >= 0)
    ng_sin = torch.where(pr_sin < 0)
    pr_deg = torch.acos(pr_cos)
    pr_deg_ng = -torch.acos(pr_cos) + 2 * math.pi
    pr_deg[ng_sin] = pr_deg_ng[ng_sin]
    pr_deg = torch.rad2deg(pr_deg)
    errors = torch.stack(
        (
            torch.abs(q_gt - pr_deg),
            torch.abs(q_gt + 360.0 - pr_deg),
            torch.abs(q_gt - (pr_deg + 360.0)),
        ),
        dim=-1,
    )
    errors, _ = torch.min(errors, dim=-1)
    losses = torch.mean(errors)
    return losses


def quaternion_loss(q_gt, q_pr):
    q_pr_norm = torch.sqrt(torch.sum(q_pr**2, dim=-1, keepdim=True))
    q_pr = q_pr / q_pr_norm
    pos_gt_loss = torch.abs(q_gt - q_pr).sum(dim=-1)
    neg_gt_loss = torch.abs(-q_gt - q_pr).sum(dim=-1)
    L1_loss = torch.minimum(pos_gt_loss, neg_gt_loss)
    L1_loss = L1_loss.mean()
    return L1_loss


def cal_angle_from_quatt(quaternions):
    r = R.from_quat(quaternions)
    eulers = r.as_euler("ZYX", degrees=True)
    return eulers


def regression_loss(preds, labels, task_type, mode="train", img_size=128):
    if task_type == "regression_shapenet1d":
        loss = F.l1_loss(preds, labels[..., :2])
    elif task_type == "regression_shapenet2d":
        loss = quaternion_loss(labels, preds)
    elif task_type == "regression_distractor":
        loss = F.l1_loss(preds, labels)
    elif task_type == "regression_pascal1d":
        loss = F.l1_loss(preds, labels)
    else:
        loss = F.mse_loss(preds, labels)
    if mode == "train":
        return loss
    elif mode == "eval":
        if task_type in ["regression_distractor"]:
            return np.mean([torch.dist(i[0]*img_size, i[1]*img_size, p=2).item() for i in zip(preds, labels)])
        elif task_type in ["regression_pascal1d"]:
            return F.l1_loss(preds*10, labels*10).detach().cpu().item()
        elif task_type in ["regression_shapenet1d"]:
            return degree_loss(labels, preds).item()
        elif task_type in ["regression_shapenet2d"]:
            return quaternion_loss(labels, preds).item()
        else:
            return dist_acc(dists=(labels - preds) ** 2)


def update(model, optimizer, train_x, train_y, task_type=None):
    optimizer.zero_grad()
    out = model(train_x, task_type=task_type)
    if task_type.startswith("regression"):
        loss = regression_loss(out, train_y, task_type, mode="train")
    else:
        loss = model.criterion(out, train_y)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        probs = F.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)
        if task_type == "segmentation":
            score = miou(preds, train_y)
        elif task_type.startswith("regression"):
            score = regression_loss(out, train_y, task_type, mode="eval")
        else:
            score = accuracy(preds, train_y)
    return score, loss.item(), probs.cpu().numpy(), preds.cpu().numpy()


def new_weights(model, best_weights, best_score, train_x, train_y, task_type=None):
    with torch.no_grad():
        eval_out = model(train_x)

        preds = torch.argmax(eval_out, dim=1)
        if task_type == "segmentation":
            eval_score = miou(preds, train_y)
        else:
            eval_score = accuracy(preds, train_y)

        tmp_best = max(eval_score, best_score)
        if tmp_best != best_score and not math.isnan(tmp_best):
            best_score = tmp_best
            best_weights = model.state_dict()
    return best_weights, best_score


def eval_model(model, x, y, task_type=None):
    with torch.no_grad():
        out = model(x, task_type=task_type)
        probs = F.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)

        # get score
        if task_type == "segmentation":
            score = miou(preds, y)
        elif task_type.startswith("regression"):
            score = regression_loss(out, y, task_type=task_type, mode="eval")
        else:
            score = accuracy(preds, y)
        # get loss
        if task_type.startswith("regression"):
            loss = regression_loss(out, y, task_type=task_type, mode="train").item()
        else:
            loss = model.criterion(out, y).item()
    return score, loss, probs.cpu().numpy(), preds.cpu().numpy()


def deploy_on_task(
    model,
    optimizer,
    train_x,
    train_y,
    test_x,
    test_y,
    T,
    test_batch_size,
    task_type=None,
):
    best_weights = model.state_dict()

    loss_history = list()

    for t in range(T):
        x_batch, y_batch = get_batch(train_x, train_y, test_batch_size)
        _, loss, _, _ = update(model, optimizer, x_batch, y_batch, task_type=task_type)
        loss_history.append(loss)

    if test_x is not None and test_y is not None:
        model.load_state_dict(best_weights)
        acc, loss, probs, preds = eval_model(model, test_x, test_y, task_type=task_type)
        loss_history.append(loss)
        return acc, loss_history, probs, preds


def process_cross_entropy(
    preds, targets, class_map, apply_softmax, dev, log=False, single_input=False
):
    one_hot = torch.zeros((preds.size(0), 2 * len(class_map.keys())), device=dev)
    if len(class_map.keys()) == 2:
        class_a, class_b = list(class_map.keys())
        one_hot[:, 0] = preds.view(-1)
        one_hot[:, 1] = 1 - preds.view(-1)
        if apply_softmax:
            one_hot[:, :2] = torch.softmax(one_hot[:, :2].clone(), dim=1)
        one_hot[targets == class_a, 2] = 1
        one_hot[targets == class_b, 3] = 1
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)

        outputs = one_hot[:, 2].detach().float().view(-1, 1)
        if single_input:
            if not log:
                one_hot = (one_hot[:, :2] * one_hot[:, 2:]).sum(dim=1).unsqueeze(1)
            else:
                one_hot = torch.log(
                    (one_hot[:, :2] * one_hot[:, 2:]).sum(dim=1).unsqueeze(1)
                )

    else:
        outputs = torch.zeros(targets.size(), dtype=torch.long, device=dev)
        num_classes = len(class_map.keys())
        for c, column in class_map.items():
            column = class_map[c]
            one_hot[:, column] = preds[:, column]
            one_hot[targets == c, num_classes + column] = 1
            outputs[targets == c] = column
        if apply_softmax:
            one_hot[:, :num_classes] = torch.softmax(
                one_hot[:, :num_classes].clone(), dim=1
            )
        if log and not single_input:
            one_hot = torch.log(one_hot + 1e-5)
        if single_input:
            if not log:
                one_hot = (
                    (one_hot[:, :num_classes] * one_hot[:, num_classes:])
                    .sum(dim=1)
                    .unsqueeze(1)
                )
            else:
                one_hot = torch.log(
                    (one_hot[:, :num_classes] * one_hot[:, num_classes:])
                    .sum(dim=1)
                    .unsqueeze(1)
                )

    return one_hot, outputs


def get_loss_and_grads(
    model,
    train_x,
    train_y,
    flat=True,
    weights=None,
    item_loss=True,
    create_graph=False,
    retain_graph=False,
    rt_only_loss=False,
    meta_loss=False,
    class_map=None,
    loss_net=None,
    loss_params=None,
    task_type=None,
):
    model.zero_grad()
    if weights is None:
        weights = model.parameters()
        out = model(train_x, task_type=task_type)
    else:
        out = model.forward_weights(train_x, weights, task_type=task_type)

    if not meta_loss:
        # Need to use L2 loss instead of CrossEntropy for regression
        if task_type.startswith("regression"):
            loss = regression_loss(out, train_y, task_type, mode="train")
        else:
            loss = model.criterion(out, train_y)
    else:
        meta_inputs, targets = process_cross_entropy(
            out, train_y, class_map=class_map, apply_softmax=True, dev=model.dev
        )
        loss = loss_net(meta_inputs, weights=loss_params)

    if rt_only_loss:
        return loss, None

    grads = torch.autograd.grad(
        loss, weights, create_graph=create_graph, retain_graph=retain_graph
    )

    if flat:
        gradients = torch.cat([p.reshape(-1) for p in grads])
        loss = torch.zeros(gradients.size()).to(train_x.device) + loss.item()
    else:
        gradients = list(grads)
        if item_loss:
            loss = loss.item()
    return loss, gradients


def put_on_device(dev, tensors):
    for i in range(len(tensors)):
        if not tensors[i] is None:
            tensors[i] = tensors[i].to(dev)
    return tensors


@contextlib.contextmanager
def empty_context():
    yield None
