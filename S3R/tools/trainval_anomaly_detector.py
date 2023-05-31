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
from anomaly.datasets.video_dataset import Dataset
from config import Config
from anomaly.utilities import PixelBar
from anomaly.engine import do_train, inference, inference2
from anomaly.models.detectors.detector import S3R
from anomaly.models.MGFN.models.mgfn import mgfn
from anomaly.engine.rtfm_model import Model as RTFM
from sklearn.ensemble import RandomForestClassifier
from anomaly.models.MGFN.train import train as trainer
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
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

from typing import Dict, List, Optional, Tuple, Union

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def fixation(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = S3RArgumentParser().parse_args()

    if args.workers != 0:
        torch.multiprocessing.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True

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
    # envcols = config.envcols

    seed = cfg.random_state
    if args.seed > 0:
        seed = args.seed
    fixation(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    global_rank = get_rank()
    logger = setup_logger("AnomalyDetection", args.log_path.joinpath(args.version), global_rank)
    # logger.info('Arguments \n{info}\n{sep}'.format(info=args, sep='-' * envcols))
    tb_logger = Logger(args.log_path.joinpath(args.version))

    env_info = get_pretty_env_info()

    # logger.info("Collecting env info (might take some time)")
    # logger.info(f"Environment Information\n\n{env_info}\n")

    data = DefaultMunch.fromDict(cfg.data)
    train_regular_dataset_cfg = data.train.regular
    train_anomaly_dataset_cfg = data.train.anomaly
    test_dataset_cfg = data.test

    split_ratio = 0.2  # usual 0.8

    # >> regular (normal) videos for the training set
    train_regular_set = Dataset(**train_regular_dataset_cfg, device=device, args=args)
    dictionary = train_regular_set.dictionary

    train_regular_idx, val_regular_idx = train_test_split(list(range(len(train_regular_set))), test_size=split_ratio)
    print(f"train_regular_idx: {len(train_regular_idx)}")
    print(f"val_regular_idx: {len(val_regular_idx)}")
    train_regular_set_model = Subset(train_regular_set, train_regular_idx)
    train_regular_set_ensemble = Subset(train_regular_set, val_regular_idx)

    # >> anomaly (abnormal) videos for the training set
    train_anomaly_dataset_cfg.dictionary = dictionary
    train_anomaly_set = Dataset(**train_anomaly_dataset_cfg, device=device, args=args)

    train_anomaly_idx, val_anomaly_idx = train_test_split(list(range(len(train_anomaly_set))), test_size=split_ratio)
    print(f"train_anomaly_idx: {len(train_anomaly_idx)}")
    print(f"val_anomaly_idx: {len(val_anomaly_idx)}")
    train_anomaly_set_model = Subset(train_anomaly_set, train_anomaly_idx)
    train_anomaly_set_ensemble = Subset(train_anomaly_set, val_anomaly_idx)

    # >> testing set
    test_dataset_cfg.dictionary = dictionary
    test_set = Dataset(**test_dataset_cfg)

    # set batch size
    args.batch_size = [args.batch_size]

    # args.batch_size.append(round(args.batch_size[0] * (len(train_anomaly_set_model) / len(train_regular_set_model))))
    args.batch_size.append(args.batch_size[0])

    train_regular_loader_m = DataLoader(
        train_regular_set_model,
        batch_size=args.batch_size[0],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator(device="cuda"),
    )
    train_regular_loader_e = DataLoader(
        train_regular_set_ensemble,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    # drop_last=True)

    train_anomaly_loader_m = DataLoader(
        train_anomaly_set_model,
        batch_size=args.batch_size[1],
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator(device="cuda"),
    )
    train_anomaly_loader_e = DataLoader(
        train_anomaly_set_ensemble,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )
    # drop_last=True)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print("**************************************************************")
    print("regular", len(train_regular_set_model), len(train_regular_set_ensemble))
    print("anomaly", len(train_anomaly_set_model), len(train_anomaly_set_ensemble))
    print("test", len(test_set))
    print("batch_size (regular, anomaly)", args.batch_size[0], args.batch_size[1])

    if args.model_name == "mgfn":
        model = mgfn(batch_size=args.batch_size[0])
    elif args.model_name == "RF":
        model = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=32, verbose=1)
    elif args.model_name == "RTFM":
        model = RTFM(args.feature_size, args.batch_size[0])
        print(
            """
        load RTFM
        """
        )
    elif args.model_name == "SVM":
        if torch.cuda.is_available():
            from cuml.svm import SVC

            print("cuml.svm.SVC")
        else:
            from sklearn.svm import SVC

            print("sklearn.svm.SVC")
        model = SVC(kernel="rbf", probability=True, verbose=3)
    else:
        model = S3R(
            args.feature_size, args.batch_size[0], args.quantize_size, dropout=args.dropout, modality=cfg.modality
        )

    # logger.info(train_regular_set.data_info)
    # logger.info(train_anomaly_set.data_info)
    # logger.info(test_set.data_info)
    # logger.info('Model Structure: \n{}'.format(model))

    # for name, value in model.named_parameters():
    #     print(name)
    if args.model_name not in ["RF", "SVM"]:
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)
        # optimizer = optim.SGD( #sgd-0.1
        #     model.parameters(),
        #     lr=config.lr[0],
        #     momentum=0.9,
        #     weiht_decay=0.005,)
        # optimizer = optim.SGD( #sgd-0.2
        #     model.parameters(),
        #     lr=config.lr[0],
        #     momentum=0.9,
        #     weight_decay=0.005,
        #     nesterov = True)
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
                score = inference2(test_loader, train_regular_loader_e, train_anomaly_loader_e, model, args, device)
            elif args.model_name != "RF":
                score, cm = inference(test_loader, model, args, device)

            # >> Performance
            title = [["Dataset", "Method", "Feature", "AUC (%)"]]
            auc = [[args.dataset, args.model_name.upper(), args.backbone.upper(), f"{score*100.:.3f}"]]

            table = AsciiTable(title + auc, " Performance on {} ".format(args.dataset))
            for i in range(len(title[0])):
                table.justify_columns[i] = "center"
            logger.info("Summary Result on {} metric\n{}".format("AUC", table.table))
            print(cm)
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.heatmap(cm, annot=True, cmap="Blues")
            plt.savefig("sklearn_confusion_matrix_annot_blues.png")

        if args.inference:
            exit()
    else:
        if args.model_name not in ["RF", "SVM"]:
            score, cm = inference(test_loader, model, args, device)

    # free Dataset
    del train_regular_loader_e
    del train_anomaly_loader_e

    if args.model_name not in ["RF", "SVM"]:
        sys_info = """
        {title}
            - dataset:\t {dataset}
            - version:\t {ver}
            - description:\t {descr}
            - initial {metric} score: {score:.3f} %
            - initial learning rate: {lr:.4f}
        """.format(
            title=color("Video Anomaly Detection", "magenta"),
            dataset=color(args.dataset, "white", attrs=["bold", "underline"]),
            ver=args.version,
            descr=color(" ".join(args.descr)),
            metric="AUC" if "xd-violence" not in args.dataset else "AP",
            score=score * 100.0,
            lr=config.lr[0],
        )

        logger.info(sys_info)

        checkpoint_filename = "{data}_{model}_i3d_best.pth"

    else:
        sys_info = """
        {title}
            - dataset:\t {dataset}
            - version:\t {ver}
            - description:\t {descr}
            - initial learning rate: {lr:.4f}
        """.format(
            title=color("Video Anomaly Detection", "magenta"),
            dataset=color(args.dataset, "white", attrs=["bold", "underline"]),
            ver=args.version,
            descr=color(" ".join(args.descr)),
            metric="AUC" if "xd-violence" not in args.dataset else "AP",
            lr=config.lr[0],
        )

        logger.info(sys_info)
        checkpoint_filename = "{data}_{model}_i3d_best.sav"
    log_filename = "{data}_{model}_i3d.score"

    # >> write title
    filename = log_filename.format(data=args.dataset, model=args.model_name)
    log_filepath = args.log_path.joinpath(args.version).joinpath(filename)
    if os.path.exists(log_filepath):
        os.remove(log_filepath)

    metric = "{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP")
    with open(log_filepath, "a") as f:
        f.write(
            "\n{sep}\n{info}\n\n\n{env}\n{sep}\n".format(
                sep="*" * 10, info=yaml.dump(args, sort_keys=False, default_flow_style=False), env=env_info
            )
        )
        f.write("\n{sep}\n{info}\n{sep}\n".format(sep="=" * 10, info=model))
        f.write("\n{}\n".format(sys_info))

        title = "| {:^6s} | {:^8s} | {:^15s} | {:^30s} | {:^30s} |".format(
            "Step", metric, "Training loss", "Elapsed time", "Now"
        )
        f.write("+{sep}+\n".format(sep="-" * (len(title) - 2)))
        f.write("{}\n".format(title))
        f.write("{sep}\n".format(sep="-" * len(title)))

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
    print(len(train_regular_loader_m), len(train_anomaly_loader_m))

    # balance = [len(train_regular_set_model), len(train_anomaly_set_model)]
    balance = [len(train_anomaly_loader_m), len(train_regular_loader_m)]
    print(balance)
    beta = 0.99
    weight = (1.0 - beta) / (1.0 - torch.pow(beta, torch.tensor(balance)))
    weights = weight / torch.sum(weight) * len(balance)
    print(weights)
    if args.model_name not in ["RF", "SVM"]:
        for step in range(last_epoch, args.max_epoch + 1):
            if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr[step - 1]

            if (step - 1) % len(train_regular_loader_m) == 0:
                loadern_iter = iter(train_regular_loader_m)

            if (step - 1) % len(train_anomaly_loader_m) == 0:
                loadera_iter = iter(train_anomaly_loader_m)
            # if (step - 1) % min(len(train_regular_loader_m), len(train_anomaly_loader_m)) == 0:
            #     loadern_iter = iter(train_regular_loader_m)
            #     loadera_iter = iter(train_anomaly_loader_m)
            if args.model_name == "mgfn":
                loss = trainer(loadern_iter, loadera_iter, balance, model, args.batch_size[0], optimizer, device)
            else:
                loss = do_train(loadern_iter, loadera_iter, balance, model, args.batch_size, optimizer, device, args)

            condition = (
                (step % 1 == 0) if args.debug else (step % args.evaluate_freq == 0 and step > args.evaluate_min_step)
            )

            if condition:
                score, cm = inference(test_loader, model, args, device)
                test_info["epoch"].append(step)
                test_info["test_{metric}".format(metric="AUC" if "xd-violence" not in args.dataset else "AP")].append(
                    score
                )
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
            ["{}".format(int(step)), "{:.3f}".format(score * 100.0), "{:^4s}".format("*" if idx == best_epoch else "")]
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
        import pickle
        from tqdm import tqdm
        import warnings
        from joblib import dump, load

        warnings.filterwarnings("ignore", category=FutureWarning)

        with torch.set_grad_enabled(True):
            # print(len(train_regular_loader_m))
            for i in tqdm(range(len(train_regular_loader_m))):
                if i % len(train_regular_loader_m) == 0:
                    loadern_iter = iter(train_regular_loader_m)
                if i % len(train_anomaly_loader_m) == 0:
                    loadera_iter = iter(train_anomaly_loader_m)
                regular_video, regular_label, macro_video, macro_label = next(loadern_iter)
                anomaly_video, anomaly_label, macro_video, macro_label = next(loadera_iter)
                # regular_video = regular_video.view(args.batch_size*32, -1)
                # anomaly_video = anomaly_video.view(args.batch_size*32, -1)
                regular_video = regular_video.view(-1, 20480)
                anomaly_video = anomaly_video.view(-1, 20480)
                video = torch.cat((regular_video, anomaly_video), 0)
                rlabel = regular_label[0 : args.batch_size].repeat(32)
                alabel = anomaly_label[0 : args.batch_size].repeat(32)
                if i == 0:
                    X_train = video.to("cpu").detach().numpy().copy()
                    y_train = torch.cat((rlabel, alabel), 0).to("cpu").detach().numpy().copy()
                else:
                    X_train = np.concatenate([X_train, video.to("cpu").detach().numpy().copy()])
                    y_train = np.concatenate(
                        [y_train, torch.cat((rlabel, alabel), 0).to("cpu").detach().numpy().copy()]
                    )

        print(f"\nstart {args.model_name}")
        if torch.cuda.is_available():
            # import cupy as cp

            # X_train = cp.asarray(X_train, dtype=cp.float32)
            # y_train = cp.asarray(y_train, dtype=cp.float32)
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)

        filename = checkpoint_filename.format(data=args.dataset, model=args.model_name)
        model.fit(X_train, y_train)
        print("finish fitting")
        pickle.dump(model, open(args.checkpoint_path.joinpath(args.version).joinpath(filename), "wb"))
        print("finish dumping")


if __name__ == "__main__":
    main()
