import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import L1Loss, MSELoss, Sigmoid
from anomaly.losses import SigmoidMAELoss, sparsity_loss, smooth_loss
from .inference2 import RandomFeatureMapDropout
from .inference2 import AddGaussianNoise


class RTFM_loss(nn.Module):
    def __init__(self, alpha, margin, weights):
        super(RTFM_loss, self).__init__()

        self.alpha = alpha
        self.margin = margin
        self.sigmoid = nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.criterion = nn.BCELoss(weight=weights)
        self.weights = weights

    def forward(
        self,
        regular_score,  # (regular) snippet-level classification score of shape Bx1
        anomaly_score,  # (anomaly) snippet-level classification score of shape Bx1
        regular_label,  # (regular) video-level label of shape B
        anomaly_label,  # (anomaly) video-level label of shape B
        regular_crest,  # (regular) selected top snippet features of shape (Bxn)xtopkxC
        anomaly_crest,  # (anomaly) selected top snippet features of shape (Bxn)xtopkxC
    ):
        label = torch.cat((regular_label, anomaly_label), 0)
        anomaly_score = anomaly_score
        regular_score = regular_score

        score = torch.cat((regular_score, anomaly_score), 0)
        score = score.squeeze()

        label = label.cuda()

        # change loss_r and loss_a to mean because of the batch size
        loss_cls = self.criterion(score, label)  # BCE loss in the score space
        loss_regular = torch.norm(torch.mean(regular_crest, dim=1), p=2, dim=1).mean()  # Txn
        loss_anomaly = torch.abs(self.margin - torch.norm(torch.mean(anomaly_crest, dim=1), p=2, dim=1)).mean()  # Txn
        loss = (self.weights[0] * loss_regular + self.weights[-1] * loss_anomaly) ** 2
        #loss = (loss_regular + loss_anomaly) ** 2

        loss_total = loss_cls + self.alpha * loss

        return loss_total


class MacroLoss(nn.Module):
    def __init__(self, weights):
        super(MacroLoss, self).__init__()

        self.loss = nn.BCELoss(weight=weights)

    def forward(self, input, label_r, label_a):
        input = input.squeeze()
        target = torch.cat((label_r, label_a), dim=0).cuda()

        loss = self.loss(input, target)

        return loss


def do_train(regular_loader, anomaly_loader, balance, model, batch_size, optimizer, device, args):
    with torch.set_grad_enabled(True):
        regular_batch_size = batch_size[0]
        anomaly_batch_size = batch_size[1]

        # random_feature_map_dropout = RandomFeatureMapDropout(p=0.5)
        # add_gaussian_noise = AddGaussianNoise(mean=0, std=0.1)
        model.train()

        """
        :param regular_video, anomaly_video
            - size: [bs, n=10, t=32, c=2048]
        """
        regular_video, regular_label, macro_r_video, macro_label_r = next(regular_loader)
        anomaly_video, anomaly_label, macro_a_video, macro_label_a = next(anomaly_loader)

        video = torch.cat((regular_video, anomaly_video), 0).to(device)
        # transform
        # transformed_features = random_feature_map_dropout(video)
        # video = add_gaussian_noise(video)

        if args.model_name != "RTFM":
            macro = torch.cat((macro_r_video, macro_a_video), 0).to(device)

            outputs = model(video, macro)
            # >> parse outputs
            anomaly_score = outputs["anomaly_score"]
            regular_score = outputs["regular_score"]
            anomaly_crest = outputs["feature_select_anomaly"]
            regular_crest = outputs["feature_select_regular"]
            video_scores = outputs["video_scores"]
            macro_scores = outputs["macro_scores"]
        else:
            outputs = model(video)  # b*32  x 2048
            anomaly_score = outputs["anomaly_score"]
            regular_score = outputs["regular_score"]
            anomaly_crest = outputs["feature_select_anomaly"]
            regular_crest = outputs["feature_select_regular"]
            video_scores = outputs["scores"]

        video_scores = video_scores.view(sum(batch_size) * 32, -1)
        beta = 0.99
        weight = (1. - beta) / (1. - torch.pow(beta, torch.tensor(balance)))
        weights = weight / torch.sum(weight) * len(balance)
        weights = torch.tensor([weights[0]] * batch_size[0] + [weights[1]] * batch_size[1]).cuda()
        # weights = torch.tensor([1.0] * batch_size[0] + [batch_size[0] / batch_size[1]] * batch_size[1]).cuda()
        # weights = torch.tensor(
        #     [batch_size[1] / sum(batch_size) * 2] * batch_size[0]
        #     + [batch_size[0] / sum(batch_size) * 2] * batch_size[1]
        # ).cuda()
        # weights = torch.tensor([1.0] * batch_size[0] + [1.0] * batch_size[1]).cuda()

        video_scores = video_scores.squeeze()
        abn_scores = video_scores[anomaly_batch_size * 32 :]

        regular_label = regular_label[0:regular_batch_size]
        anomaly_label = anomaly_label[0:anomaly_batch_size]
        loss_criterion = RTFM_loss(0.0001, 100, weights)
        loss_magnitude = loss_criterion(
            regular_score, anomaly_score, regular_label, anomaly_label, regular_crest, anomaly_crest
        )

        loss_sparse = sparsity_loss(abn_scores, 8e-3)
        loss_smooth = smooth_loss(abn_scores, 8e-4)

        if args.model_name != "RTFM":
            macro_criterion = MacroLoss(weights)
            loss_macro = macro_criterion(macro_scores, macro_label_r, macro_label_a)

            cost = loss_magnitude + loss_smooth + loss_sparse + loss_macro
        else:
            cost = loss_magnitude + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    return cost
