import torch
import numpy as np


from torchmetrics.classification import AUROC

from sklearn.metrics import auc, roc_curve


def inference(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        # model = copy.deepcopy(model).half()
        pred = torch.zeros(0).to(device)

        gt = dataloader.dataset.ground_truths
        # dataset = args.dataset.lower()

        # if args.inference:
        #     video_list = dataloader.dataset.video_list
        #     result_dict = dict()
        for video, macro in dataloader:
            video = video.to(device)  # video.half().to(device)
            video = video.permute(0, 2, 1, 3)

            if args.model_name not in ["mgfn", "RTFM"]:
                macro = macro.to(device)  # macro.half().to(device)
                outputs = model(video, macro)
                logits = outputs["video_scores"]
            elif args.model_name == "RTFM":
                outputs = model(video)
                logits = outputs["scores"]
            else:
                _, _, _, _, logits = model(video)

            # >> parse outputs
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            # if args.inference:
            #     video_id = video_list[i]
            #     result_dict[video_id] = logits.cpu().detach().numpy()

            sig = logits
            pred = torch.cat((pred, sig))
        # if args.inference:
        #     out_dir = f'output/{dataset}'
        #     import pickle
        #     with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
        #         pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        # pred = pred.repeat_interleave(16)
        # metric = AUROC(task="binary")
        # score = metric(pred, torch.from_numpy(gt).to(device))

        pred = np.array(pred.cpu().detach().numpy())
        pred = np.repeat(pred, 16)

        fpr, tpr, threshold = roc_curve(gt, pred)
        rec_auc = auc(fpr, tpr)
        score = rec_auc

        return score
