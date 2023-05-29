import os
import sys
import yaml
import time
import json
import torch
import random
import datetime
import numpy as np
import torch.optim as optim

import _init_paths

from tqdm import tqdm
from munch import DefaultMunch
from utils import save_best_record
from torch.utils.data import DataLoader
from terminaltables import AsciiTable, DoubleTable
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data.dataset import Subset

from config import Config
from anomaly.utilities import PixelBar
from anomaly.engine import do_train, inference, inference2
from anomaly.datasets.video_dataset import Dataset
from anomaly.models.detectors.detector import S3R
from anomaly.models.MGFN.models.mgfn import mgfn
from anomaly.engine.rtfm_model import Model as RTFM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from anomaly.models.MGFN.train import train as trainer
import torchvision.transforms as transforms
from anomaly.apis import (
    mkdir,
    color,
    AverageMeter,
    setup_logger,
    setup_tblogger,
    Logger,
    synchronize,
    get_rank,
    S3RArgumentParser,
)
from sklearn.model_selection import StratifiedKFold

from typing import Dict, List, Optional, Tuple, Union


def fixation(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_inference(
    test_loader,
    model,
    logger,
    args,
    device,
):
    if args.resume:
        if os.path.isfile(args.resume):
            print("A")
            if args.model_name != "RF":
                checkpoint = torch.load(args.resume)

                model.load_state_dict(checkpoint)
            else:
                model = pickle.load(open(args.resume, "rb"))
            print(">>> Checkpoint {} loaded.".format(color(args.resume)))

        if args.inference:
            exit()

    return model


def main():
    args = S3RArgumentParser().parse_args()

    if not args.inference:
        mkdir(args.checkpoint_path.joinpath(args.version))
        mkdir(args.log_path.joinpath(args.version))
        mkdir(args.dictionary_path)

    if "shanghaitech" in args.dataset:
        import configs.shanghaitech.shanghaitech_dl as cfg
    elif "ucf-crime" in args.dataset:
        import configs.ucf_crime.ucf_crime_dl as cfg

    if args.lr:
        lr = [args.lr] * args.max_epoch
    else:
        lr = [cfg.init_lr] * args.max_epoch

    config = Config(lr)
    envcols = config.envcols

    seed = cfg.random_state
    if args.seed > 0:
        seed = args.seed
    fixation(seed)

    global_rank = get_rank()
    logger = setup_logger("AnomalyDetection", args.log_path.joinpath(args.version), global_rank)
    # logger.info('Arguments \n{info}\n{sep}'.format(info=args, sep='-' * envcols))
    tb_logger = Logger(args.log_path.joinpath(args.version))

    env_info = get_pretty_env_info()

    # logger.info("Collecting env info (might take some time)")
    # logger.info(f"Environment Information\n\n{env_info}\n")

    ##############
    # data load
    ##############
    data = DefaultMunch.fromDict(cfg.data)
    train_regular_dataset_cfg = data.train.regular
    train_anomaly_dataset_cfg = data.train.anomaly
    test_dataset_cfg = data.test

    split_ratio = 0.75  # usual 0.8
    # >> regular (normal) videos for the training set
    train_regular_set = Dataset(**train_regular_dataset_cfg)
    dictionary = train_regular_set.dictionary

    len_train_regular_m = int(len(train_regular_set) * split_ratio)
    train_regular_set_model = Subset(train_regular_set, range(len_train_regular_m))
    train_regular_set_ensemble = Subset(train_regular_set, range(len_train_regular_m, len(train_regular_set)))

    # >> anomaly (abnormal) videos for the training set
    train_anomaly_dataset_cfg.dictionary = dictionary
    train_anomaly_set = Dataset(**train_anomaly_dataset_cfg)

    len_train_anomaly_m = int(len(train_anomaly_set) * split_ratio)
    train_anomaly_set_model = Subset(train_anomaly_set, range(len_train_anomaly_m))
    train_anomaly_set_ensemble = Subset(train_anomaly_set, range(len_train_anomaly_m, len(train_anomaly_set)))

    # >> testing set
    test_dataset_cfg.dictionary = dictionary
    test_set = Dataset(**test_dataset_cfg)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False)

    if args.model_name == "mgfn":
        model = mgfn()
    elif args.model_name == "RF":
        model = RandomForestClassifier(n_estimators=1000, random_state=0)
    elif args.model_name == "RTFM":
        model = RTFM(args.feature_size, args.batch_size)
    elif args.model_name == "SVM":
        model = SVC(probability=True)
    else:
        model = S3R(args.feature_size, args.batch_size, args.quantize_size, dropout=args.dropout, modality=cfg.modality)

    if args.model_name not in ["RF", "SVM"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)
    test_info = {
        "epoch": [],
        "elapsed": [],
        "now": [],
        "train_loss": [],
        "test_{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP"): [],
    }
    best_AUC = -1

    ###added
    global_ite_idx = 0
    ###
    last_epoch = 1

    # inference
    model = run_inference(test_loader, model, logger, args, device)

    if args.model_name not in ["RF", "SVM"]:
        checkpoint_filename = "{data}_{model}_i3d_best.pth"

    else:
        checkpoint_filename = "{data}_{model}_i3d_best.sav"
    log_filename = "{data}_{model}_i3d.score"

    # >> write title
    filename = log_filename.format(data=args.dataset, model=args.model_name)
    log_filepath = args.log_path.joinpath(args.version).joinpath(filename)
    if os.path.exists(log_filepath):
        os.remove(log_filepath)

    metric = "{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP")

    process_time = AverageMeter()
    end = time.time()
    bar = PixelBar(
        "{now} - {dataset} - INFO -".format(
            now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            dataset=color(args.dataset),
        ),
        max=args.max_epoch,
    )

    start_time = time.time()

    statistics = []
    if args.debug:
        args.max_epoch = 3

    regular_indices = list(range(len(train_regular_set_model)))
    anomaly_indices = list(range(len(train_anomaly_set_model)))
    # 層化k分割交差検証
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    meta_X = np.zeros((len(regular_indices) + len(anomaly_indices), n_base_learners))
    meta_y = np.zeros(len(regular_indices) + len(anomaly_indices))
    # 学習器の数
    n_base_learners = 5  # S3R, MGFN, RTFM, Random Forest, SVM

    if args.model_name not in ["RF", "SVM"]:
        for train_idx, val_idx in skf.split(np.concatenate((regular_indices, anomaly_indices)), y):
            regular_train_idx = np.intersect1d(train_idx, regular_indices)
            regular_val_idx = np.intersect1d(val_idx, regular_indices)
            anomaly_train_idx = np.intersect1d(train_idx, anomaly_indices)
            anomaly_val_idx = np.intersect1d(val_idx, anomaly_indices)

            for step in range(last_epoch, args.max_epoch + 1):
                if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = config.lr[step - 1]

                if (step - 1) % len(train_regular_loader_m) == 0:
                    loadern_iter = iter(train_regular_loader_m)

                if (step - 1) % len(train_anomaly_loader_m) == 0:
                    loadera_iter = iter(train_anomaly_loader_m)

                if args.model_name == "mgfn":
                    loss = trainer(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device)
                else:
                    loss = do_train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, device, args)

                condition = (
                    (step % 1 == 0)
                    if args.debug
                    else (step % args.evaluate_freq == 0 and step > args.evaluate_min_step)
                )

                if condition:
                    score = inference(test_loader, model, args, device)

                    test_info["epoch"].append(step)
                    test_info[
                        "test_{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP")
                    ].append(score)
                    test_info["train_loss"].append(loss)

                    test_info["elapsed"].append(str(datetime.timedelta(seconds=time.time() - start_time)))
                    now = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    test_info["now"].append(now)

                    statistics.append([step, score])

                    metric = "test_{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP")
                    if test_info[metric][-1] > best_AUC:
                        best_AUC = test_info[metric][-1]
                        filename = checkpoint_filename.format(data=args.dataset, model=args.model_name)
                        torch.save(model.state_dict(), args.checkpoint_path.joinpath(args.version).joinpath(filename))

                        save_best_record(test_info, log_filepath, metric)

                # measure elapsed time
                process_time.update(time.time() - end)
                end = time.time()

                ######
                # added
                #
                # tensorboard logger
                tb_info = {
                    "loss": loss,
                    "AUC": score,
                }
                for tag, value in tb_info.items():
                    tb_logger.scalar_summary(tag, value, step)

                #
                ######

                # plot progress
                info = (
                    "({cnt}/{num})"
                    " time: {pt:.3f}s, total: {total:}, eta: {eta:},"
                    " lr: {lr}, loss: {loss:.4f}, {metric}: {score:.3f}".format(
                        cnt=step,
                        num=args.max_epoch,
                        pt=process_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        lr=optimizer.param_groups[0]["lr"],
                        loss=loss,
                        metric="AUC" if "xd-violence" not in args.dataset else "AP",
                        score=score * 100.0,
                    )
                )

                bar.suffix = info
                bar.next()

            bar.finish()
            with open(log_filepath, "a") as f:
                f.write("+{sep}+\n".format(sep="-" * (len(title) - 2)))

            # Performance
            auc_performance = np.array(statistics)
            best_epoch = np.argmax(auc_performance[:, -1])
            title = [["Step", "AUC", "Best"]] if "xd-violence" not in args.dataset else [["Step", "AP", "Best"]]
            score = [
                [
                    "{}".format(int(step)),
                    "{:.3f}".format(score * 100.0),
                    "{:^4s}".format("*" if idx == best_epoch else ""),
                ]
                for idx, (step, score) in enumerate(auc_performance)
            ]

            # show top-k scores
            performance = np.array(score)[:, 1].astype(np.float32)
            top_k_idx = np.argsort(performance)  # ascending order
            top_k_idx = top_k_idx[-args.report_k :]  # only show k scores
            score = np.array(score)[top_k_idx].tolist()

            table = AsciiTable(title + score, " Performance on {} ".format(args.dataset))
            table.justify_columns[0], table.justify_columns[-1] = "center", "center"
            logger.info("Summary Result on {} metric\n{}".format("AUC", table.table))
    else:
        with torch.set_grad_enabled(True):
            # print(len(train_regular_loader_m))
            for i in range(len(train_regular_loader_m)):
                # If we have finished a full epoch, we need to reset the data loaders.
                if i % len(train_regular_loader_m) == 0:
                    loadern_iter = iter(train_regular_loader_m)
                if i % len(train_anomaly_loader_m) == 0:
                    loadera_iter = iter(train_anomaly_loader_m)
                # Get the next batch of data.
                regular_video, regular_label, macro_video, macro_label = next(loadern_iter)
                anomaly_video, anomaly_label, macro_video, macro_label = next(loadera_iter)
                # regular_video = regular_video.view(args.batch_size*32, -1)
                # anomaly_video = anomaly_video.view(args.batch_size*32, -1)
                regular_video = regular_video.view(-1, 20480)
                anomaly_video = anomaly_video.view(-1, 20480)
                video = torch.cat((regular_video, anomaly_video), 0)
                rlabel = regular_label[0 : args.batch_size].repeat(32)
                alabel = anomaly_label[0 : args.batch_size].repeat(32)
                # Concatenate the regular and anomaly batches.
                if i == 0:
                    X_train = video.to("cpu").detach().numpy().copy()
                    y_train = torch.cat((rlabel, alabel), 0).to("cpu").detach().numpy().copy()
                else:
                    X_train = np.concatenate([X_train, video.to("cpu").detach().numpy().copy()])
                    y_train = np.concatenate(
                        [y_train, torch.cat((rlabel, alabel), 0).to("cpu").detach().numpy().copy()]
                    )

        print(f"\nstart {args.model_name}")
        filename = checkpoint_filename.format(data=args.dataset, model=args.model_name)
        model.fit(X_train, y_train)
        pickle.dump(model, open(args.checkpoint_path.joinpath(args.version).joinpath(filename), "wb"))


if __name__ == "__main__":
    main()
