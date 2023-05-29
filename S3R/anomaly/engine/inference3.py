import torch
from torch import nn
import numpy as np
import os.path as osp
import sys
sys.path.append('../../S3R')
from ..models.MGFN import mgfn_model
from .rtfm_load import rtfm_model
import pickle


#torch.set_default_tensor_type("torch.cuda.FloatTensor")

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from sklearn.linear_model import LogisticRegression


from sklearn.model_selection import train_test_split




def inference(dataloader, s3r, args, device):
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

        mgfn = mgfn_model().get(device, args.resume2)
        rtfm = rtfm_model(device, args.resume3)
        
        RF_model = pickle.load(open(args.resume4, 'rb'))
        SVM_model = pickle.load(open(args.resume5, 'rb'))
        
        
        #liblinear 97.566
        s3r.eval()
        mgfn.eval()
        rtfm.eval()
        y_label = []
        
        meta_model = pickle.load(open("checkpoint/meta_model.sav", 'rb'))
        
        
        for i, (video, macro) in enumerate(dataloader):
            video = video.to(device)
            video = video.permute(0, 2, 1, 3)
            
            shape = video.shape
            #print(shape)

            macro = macro.to(device)
            outputs = s3r(video, macro)
            _, _, _, _, logits_mgfn = mgfn(video)
            outputs_rtfm = rtfm(video)
            #RF
            sig_RF = RF_model.predict_proba(video.reshape(-1, 20480).to('cpu').detach().numpy().copy())
            sig_RF = sig_RF[:,1]
            #SVM
            sig_SVM = SVM_model.predict_proba(video.reshape(-1, 20480).to('cpu').detach().numpy().copy())
            sig_SVM = sig_SVM[:,1]
                

            # >> parse outputs
            #s3r
            logits = outputs['video_scores']
            #print(video.shape, logits.shape)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            
            #rint(video.shape, logits.shape)
            
            #mgfn
            logits_mgfn = torch.squeeze(logits_mgfn, 1)
            logits_mgfn = torch.mean(logits_mgfn, 0)
            
            #rtfm
            logits_rtfm = outputs_rtfm['scores']

            logits_rtfm = torch.squeeze(logits_rtfm, 1)
            logits_rtfm = torch.mean(logits_rtfm, 0)
            

            if args.inference:
                video_id = video_list[i]
                result_dict[video_id] = logits.cpu().detach().numpy()
            
            #s3r
            sig = logits
            pred = torch.cat((pred, sig))
            
            #mgfn
            sig_mgfn = logits_mgfn
            pred_mgfn = torch.cat((pred_mgfn, sig_mgfn)) 
            
            #rtfm
            sig_rtfm = logits_rtfm
            pred_rtfm = torch.cat((pred_rtfm, sig_rtfm))
            
            #RF
            pred_RF = np.append(pred_RF, sig_RF)
            
            #SVM
            pred_SVM = np.append(pred_SVM, sig_SVM)
        # ===== mean ======
        # m = nn.Sigmoid()
        # pred_total = m(torch.mean(torch.stack([pred, pred_mgfn, pred_rtfm]),dim=0)) #
        # print(pred_total.shape)
        print(pred.shape, pred_mgfn.shape, pred_rtfm.shape, len(pred_RF), len(pred_SVM))
        stack_pred = np.column_stack((np.repeat(np.array(pred.cpu().detach().numpy()), 16)
                                        ,np.repeat(np.array(pred_mgfn.cpu().detach().numpy()), 16)
                                        ,np.repeat(np.array(pred_rtfm.cpu().detach().numpy()), 16)
                                        ,np.repeat(pred_RF,16)
                                        ,np.repeat(pred_SVM,16)
                                     )
                                    )
        m = nn.Sigmoid()
        pred_total = m(torch.mean(torch.stack([pred, pred_mgfn, pred_rtfm,  torch.unsqueeze(torch.from_numpy(pred_SVM).to(device),dim=1)]),dim=0)) #torch.unsqueeze(torch.from_numpy(pred_RF).to(device), dim=1),
        meta_test_pred = meta_model.predict_proba(stack_pred)
        
        

        if args.inference:
            out_dir = f'output/{dataset}'
            with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
                pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)
                
                
        pred, pred_mgfn, pred_rtfm, pred_RF, pred_SVM = stack_pred.T
        #pred, pred_mgfn, pred_rtfm = las_stack_pred.T
        # pred_s3r

        fpr, tpr, threshold = roc_curve(gt, pred)
        print(f"S3R : {auc(fpr, tpr)}")
        
        # pred_mgfn

        fpr, tpr, threshold = roc_curve(gt, pred_mgfn)
        print(f"MGFN : {auc(fpr, tpr)}")
        
        # pred_rtfm
        fpr, tpr, threshold = roc_curve(gt, pred_rtfm)
        print(f"RTFM : {auc(fpr, tpr)}")
        
        fpr, tpr, threshold = roc_curve(gt, pred_RF)
        print(f"RF : {auc(fpr, tpr)}")
        
        fpr, tpr, threshold = roc_curve(gt, pred_SVM)
        print(f"SVM : {auc(fpr, tpr)}")
        
        # pred_total
        #meta_test_pred = list(meta_test_pred.cpu().detach().numpy())
        #meta_test_pred = np.repeat(np.array(meta_test_pred), 16)
        print(meta_test_pred.shape)
        fpr, tpr, threshold = roc_curve(gt, meta_test_pred[:,1])
        rec_auc = auc(fpr, tpr)
        score = rec_auc
        
        pred_total = list(pred_total.cpu().detach().numpy())
        pred_total = np.repeat(np.array(pred_total), 16)
        fpr, tpr, threshold = roc_curve(gt, pred_total)
        rec_auc1 = auc(fpr, tpr)
        score1 = rec_auc1
        

        return score, score1