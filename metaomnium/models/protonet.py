import torch
import torch.nn.functional as F

from .learner import Learner
from .modules.utils import accuracy, empty_context, miou, put_on_device, regression_loss


class PrototypicalNetwork(Learner):
    def __init__(self, meta_batch_size=1, **kwargs):
        super().__init__(**kwargs)
        self.meta_batch_size = meta_batch_size
        self.task_counter = 0
        self.baselearner = self.baselearner_fn(**self.baselearner_args).to(self.dev)
        self.initialization = [
            p.clone().detach().to(self.dev) for p in self.baselearner.parameters()
        ]
        self.mode = "inverse_distance_weighted"
        for p in self.initialization:
            p.requires_grad = True

        if self.opt_fn == "sgd":
            self.optimizer = torch.optim.SGD(
                self.initialization, lr=self.lr, momentum=self.momentum
            )
        else:
            self.optimizer = torch.optim.Adam(self.initialization, lr=self.lr)
        self.dist_temperature = kwargs["dist_temperature"]

    def _calculate_distance(self, fts, prototype, scaler=20, task_type=None):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        if task_type == "segmentation":
            dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
        elif task_type.startswith("regression"):
            dist = torch.zeros(size=(fts.shape[0], prototype.shape[0])).cuda()
            for i, sample_query in enumerate(fts):
                dist[i] = (
                    -torch.cdist(sample_query.unsqueeze(0), prototype)
                    / self.dist_temperature
                )
        return dist

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
            contxt = empty_context
            num_classes = self.baselearner.train_classes
        else:
            contxt = torch.no_grad
            if num_classes is None:
                num_classes = self.baselearner.eval_classes

        with contxt():
            support_embeddings = self.baselearner.forward_weights(
                train_x, self.initialization, embedding=True, task_type=task_type
            )
            query_embeddings = self.baselearner.forward_weights(
                test_x, self.initialization, embedding=True, task_type=task_type
            )

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
                prototypes = [
                    bg_prototype,
                ] + fg_prototypes

                dists = [
                    self._calculate_distance(
                        query_embeddings, prototype, task_type=task_type
                    )
                    for prototype in prototypes
                ]
                pred = torch.stack(dists, dim=1)  # N x (1 + Wa) x H' x W'

                out = F.interpolate(pred, size=test_y.shape[-2:], mode="bilinear")

                loss = self.baselearner.criterion(out, test_y) / (
                    num_classes * len(test_y)
                )
            elif task_type.startswith("regression"):
                dists = self._calculate_distance(
                    query_embeddings, support_embeddings, task_type="regression"
                )
                dists_norm = torch.nn.Softmax(dim=1)(dists)
                if task_type == "regression_shapenet1d":
                    train_y = train_y[..., :2]
                out = torch.mm(dists_norm, train_y)
                if task_type == "regression_pascal1d":
                    out = out[..., :2]
                    test_y = test_y[..., :2]
                loss = regression_loss(out, test_y, task_type, mode="train")
            else:
                prototypes = torch.zeros(
                    (num_classes, support_embeddings.size(1)),
                    device=self.initialization[0].device,
                )
                for class_id in range(num_classes):
                    mask = train_y == class_id
                    prototypes[class_id] = (
                        support_embeddings[mask].sum(dim=0) / torch.sum(mask).item()
                    )

                distance_matrix = (
                    torch.cdist(query_embeddings.unsqueeze(0), prototypes.unsqueeze(0))
                    ** 2
                ).squeeze(0)
                out = -1 * distance_matrix

                loss = self.baselearner.criterion(out, test_y) / (
                    num_classes * len(test_y)
                )

        with torch.no_grad():
            if not task_type.startswith("regression"):
                probs = F.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
            if task_type == "segmentation":
                score = miou(preds, test_y)
            elif task_type.startswith("regression"):
                if task_type == "regression_pascal1d":
                    out = out[..., :2]
                    test_y = test_y[..., :2]
                score = regression_loss(out, test_y, task_type, mode="eval")
                preds = out.detach()
                probs = out.detach()
            else:
                score = accuracy(preds, test_y)

        return score, loss, probs.cpu().numpy(), preds.cpu().numpy()

    def train(self, train_x, train_y, test_x, test_y, task_type):
        self.baselearner.train()
        self.task_counter += 1
        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        score, loss, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, True, task_type=task_type
        )

        loss.backward()
        if self.task_counter % self.meta_batch_size == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return score, loss.item(), probs, preds

    def evaluate(
        self, num_classes, train_x, train_y, test_x, test_y, task_type, **kwargs
    ):
        self.baselearner.eval()
        train_x, train_y, test_x, test_y = put_on_device(
            self.dev, [train_x, train_y, test_x, test_y]
        )

        score, loss, probs, preds = self._deploy(
            train_x, train_y, test_x, test_y, False, num_classes, task_type=task_type
        )

        return score, [loss.item()], probs, preds

    def dump_state(self):
        return [p.clone().detach() for p in self.initialization]

    def load_state(self, state):
        self.initialization = [p.clone() for p in state]
        for p in self.initialization:
            p.requires_grad = True
