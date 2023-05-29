from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import pickle
from .rtfm_load import rtfm_model
from ..models.MGFN import mgfn_model
import torch
from torch import nn
import numpy as np
import os.path as osp
import sys
from scipy.stats import mode

from tqdm import tqdm

sys.path.append("../../S3R")


torch.set_default_tensor_type("torch.cuda.FloatTensor")

import random


class RandomFeatureMapDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(RandomFeatureMapDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            dropout_mask = torch.rand(x.shape[0], x.shape[1], 1, 1) > self.p
            return x * dropout_mask.to(x.device)
        else:
            return x


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.1):
        super(AddGaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn(x.size()) * self.std + self.mean
            return x + noise.to(x.device)
        else:
            return x


# class EnsembleModel(nn.Module):
#     def __init__(self, modelA, modelB, num_features):
#         super().__init__()
#         self.modelA = modelA
#         self.modelB = modelB
#         self.classifier = nn.Linear(2*num_features, num_features)

#     def forward(self, x):
#         x1 = self.modelA(x)
#         x2 = self.modelB(x)
#         x = torch.cat((x1, x2), dim=1)
#         x = self.classifier(F.relu(x))
#         return x


from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

random_feature_map_dropout = RandomFeatureMapDropout(p=0.5)
add_gaussian_noise = AddGaussianNoise(mean=0, std=0.1)


def plot_confusion_matrix(test_y, pred_y, normalize=False):
    cm = confusion_matrix(test_y, pred_y.round())
    print(cm)
    sns.heatmap(cm, square=True, cbar=True, annot=True, cmap="Blues")
    plt.xlabel("Pre", fontsize=13)
    plt.ylabel("GT", fontsize=13)
    plt.show()
    return


def run_models_for_ensemble(
    video,
    macro,
    s3r,
    mgfn,
    rtfm,
    RF_model,
    SVM_model,
    pred,
    pred_mgfn,
    pred_rtfm,
    pred_RF,
    pred_SVM,
    device,
):
    # video = video.permute(0, 2, 1, 3)

    shape = video.shape
    # print(shape)

    #     transformed_features = random_feature_map_dropout(video)
    #     video = add_gaussian_noise(transformed_features)

    macro = macro.to(device)
    outputs = s3r(video, macro)
    _, _, _, _, logits_mgfn = mgfn(video)
    outputs_rtfm = rtfm(video)
    sig_RF = RF_model.predict_proba(video.reshape(-1, 20480).to("cpu").detach().numpy().copy())
    sig_RF = sig_RF[:, 1]

    sig_SVM = SVM_model.predict_proba(video.reshape(-1, 20480).to("cpu").detach().numpy().copy())
    sig_SVM = sig_SVM[:, 1]

    # >> parse outputs
    # s3r
    logits = outputs["video_scores"]
    # print(video.shape, logits.shape)
    logits = torch.squeeze(logits, 1)
    logits = torch.mean(logits, 0)

    # rint(video.shape, logits.shape)

    # mgfn
    logits_mgfn = torch.squeeze(logits_mgfn, 1)
    logits_mgfn = torch.mean(logits_mgfn, 0)

    # rtfm
    logits_rtfm = outputs_rtfm["scores"]

    logits_rtfm = torch.squeeze(logits_rtfm, 1)
    logits_rtfm = torch.mean(logits_rtfm, 0)

    # video_id = video_list[i]
    # result_dict[video_id] = logits.cpu().detach().numpy()

    # s3r
    sig = logits
    # print(sig.shape)
    pred = torch.cat((pred, sig))

    # mgfn
    sig_mgfn = logits_mgfn
    pred_mgfn = torch.cat((pred_mgfn, sig_mgfn))

    # rtfm
    sig_rtfm = logits_rtfm
    pred_rtfm = torch.cat((pred_rtfm, sig_rtfm))

    # RF
    pred_RF = np.append(pred_RF, sig_RF)

    # SVM
    pred_SVM = np.append(pred_SVM, sig_SVM)
    return pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM


def print_each_auc(stack_pred, gt):
    stack_pred = stack_pred
    # pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM = stack_pred.T
    pred, pred_mgfn, pred_RF, pred_SVM = stack_pred.T
    # pred, pred_mgfn, pred_rtfm = las_stack_pred.T
    # pred_s3r

    print("*****Confusion Matrix : S3R Model*****")
    plot_confusion_matrix(gt, pred)

    fpr, tpr, threshold = roc_curve(gt, pred)
    print(f"S3R : {auc(fpr, tpr)}")

    # pred_mgfn
    print("*****Confusion Matrix : MGFN Model*****")
    plot_confusion_matrix(gt, pred_mgfn)

    fpr, tpr, threshold = roc_curve(gt, pred_mgfn)
    print(f"MGFN : {auc(fpr, tpr)}")

    # # pred_rtfm
    # print("*****Confusion Matrix : RTFM Model*****")
    # plot_confusion_matrix(gt, pred_rtfm)

    # fpr, tpr, threshold = roc_curve(gt, pred_rtfm)
    # print(f"RTFM : {auc(fpr, tpr)}")

    # pred_RF
    print("*****Confusion Matrix : RF Model*****")
    plot_confusion_matrix(gt, pred_RF)

    fpr, tpr, threshold = roc_curve(gt, pred_RF)
    print(f"RF : {auc(fpr, tpr)}")

    # pred_SVM
    print("*****Confusion Matrix : SVM Model*****")
    plot_confusion_matrix(gt, pred_SVM)

    fpr, tpr, threshold = roc_curve(gt, pred_SVM)
    print(f"SVM : {auc(fpr, tpr)}")


def inference(dataloader, er_dataloder, ea_dataloder, s3r, args, device):
    with torch.no_grad():
        pred = torch.zeros(0).to(device)

        gt = dataloader.dataset.ground_truths
        dataset = args.dataset.lower()

        if args.inference:
            video_list = dataloader.dataset.video_list
            result_dict = dict()

        pred_mgfn = torch.zeros(0).to(device)
        pred_rtfm = torch.zeros(0).to(device)
        pred_RF = []
        pred_SVM = []

        mgfn = mgfn_model(args.batch_size).get(device, args.resume2)
        rtfm = rtfm_model(device, args.resume3)

        RF_model = pickle.load(open(args.resume4, "rb"))
        SVM_model = pickle.load(open(args.resume5, "rb"))

        meta_model = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
        # liblinear 97.566
        s3r.eval()
        mgfn.eval()
        rtfm.eval()
        y_label = []

        # regular
        regular_loader = iter(er_dataloder)
        anomaly_loader = iter(ea_dataloder)
        for i in tqdm(range(len(er_dataloder))):
            video, label, macro, macro_label = next(regular_loader)

            label = label[0 : args.batch_size].repeat(32 * 16)
            y_label = np.append(y_label, label.to("cpu").detach().numpy().copy())
            pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM = run_models_for_ensemble(
                video,
                macro,
                s3r,
                mgfn,
                rtfm,
                RF_model,
                SVM_model,
                pred,
                pred_mgfn,
                pred_rtfm,
                pred_RF,
                pred_SVM,
                device,
            )
            # anomaly
            if i < len(ea_dataloder):
                video, label, macro, macro_label = next(anomaly_loader)

                label = label[0 : args.batch_size].repeat(32 * 16)
                y_label = np.append(y_label, label.to("cpu").detach().numpy().copy())
                pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM = run_models_for_ensemble(
                    video,
                    macro,
                    s3r,
                    mgfn,
                    rtfm,
                    RF_model,
                    SVM_model,
                    pred,
                    pred_mgfn,
                    pred_rtfm,
                    pred_RF,
                    pred_SVM,
                    device,
                )

        # print(pred.shape, pred_mgfn.shape, pred_rtfm.shape, len(pred_RF))
        # ensemble
        stack_pred = np.column_stack(
            (
                np.repeat(np.array(pred.cpu().detach().numpy()), 16),
                np.repeat(np.array(pred_mgfn.cpu().detach().numpy()), 16),
                # np.repeat(np.array(pred_rtfm.cpu().detach().numpy()), 16),
                np.repeat(pred_RF, 16),
                np.repeat(pred_SVM, 16),
            )
        )
        # print(f"stack_pred : {len(stack_pred)}, y_label : {len(y_label)}")
        filename = "meta_model.sav"
        meta_model.fit(stack_pred, y_label)

        pickle.dump(meta_model, open(args.checkpoint_path.joinpath(filename), "wb"))
        print("finish meta model fit and save")
        print_each_auc(stack_pred, y_label)

        pred = torch.zeros(0)
        pred_mgfn = torch.zeros(0)
        pred_rtfm = torch.zeros(0)
        pred_RF = []
        pred_SVM = []

        for i, (video, macro) in tqdm(enumerate(dataloader)):
            video = video.to(device)
            video = video.permute(0, 2, 1, 3)
            pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM = run_models_for_ensemble(
                video,
                macro,
                s3r,
                mgfn,
                rtfm,
                RF_model,
                SVM_model,
                pred,
                pred_mgfn,
                pred_rtfm,
                pred_RF,
                pred_SVM,
                device,
            )
        # ===== mean ======
        # m = nn.Sigmoid()
        # pred_total = m(torch.mean(torch.stack([pred, pred_mgfn, pred_rtfm]),dim=0)) #
        # print(pred_total.shape)
        print(pred.shape, pred_mgfn.shape, pred_rtfm.shape, len(pred_RF), len(pred_SVM))
        stack_pred = np.column_stack(
            (
                np.repeat(np.array(pred.cpu().detach().numpy()), 16),
                np.repeat(np.array(pred_mgfn.cpu().detach().numpy()), 16),
                # np.repeat(np.array(pred_rtfm.cpu().detach().numpy()), 16),
                np.repeat(pred_RF, 16),
                np.repeat(pred_SVM, 16),
            )
        )

        meta_test_pred = meta_model.predict_proba(stack_pred)

        if args.inference:
            out_dir = f"output/{dataset}"
            with open(osp.join(out_dir, f"{dataset}_taskaware_results.pickle"), "wb") as fout:
                pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        print_each_auc(stack_pred, gt)
        # pred_total
        # meta_test_pred = list(meta_test_pred.cpu().detach().numpy())
        # meta_test_pred = np.repeat(np.array(meta_test_pred), 16)
        print(meta_test_pred.shape)
        fpr, tpr, threshold = roc_curve(gt, meta_test_pred[:, 1])
        rec_auc = auc(fpr, tpr)
        score = rec_auc

        print("*****Confusion Matrix : Meta Model*****")
        plot_confusion_matrix(gt, (meta_test_pred[:, 1]).round())

        voted_pred = mode(stack_pred, axis=1).mode

        print("***meta model auc (vote)***")
        fpr, tpr, threshold = roc_curve(gt, voted_pred)
        rec_auc = auc(fpr, tpr)
        print("metamodel", rec_auc)

        return score
