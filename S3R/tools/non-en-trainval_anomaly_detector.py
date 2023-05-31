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
from anomaly.engine import do_train, inference, inference2, inference3
from anomaly.datasets.video_dataset import Dataset
from anomaly.models.detectors.detector import S3R
from anomaly.models.MGFN.models.mgfn import mgfn
from anomaly.engine.rtfm_model import Model as RTFM
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from anomaly.models.MGFN.train import train as trainer
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
import pickle
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
    envcols = config.envcols

    seed = cfg.random_state
    if args.seed > 0:
        seed = args.seed
    fixation(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    global_rank = get_rank()
    logger = setup_logger("AnomalyDetection", args.log_path.joinpath(args.version), global_rank)
    # logger.info("Arguments \n{info}\n{sep}".format(info=args, sep="-" * envcols))
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

    # >> regular (normal) videos for the training set
    train_regular_set = Dataset(**train_regular_dataset_cfg, device=device, args=args)
    dictionary = train_regular_set.dictionary

    # >> anomaly (abnormal) videos for the training set
    train_anomaly_dataset_cfg.dictionary = dictionary
    train_anomaly_set = Dataset(**train_anomaly_dataset_cfg, device=device, args=args)

    # >> testing set
    test_dataset_cfg.dictionary = dictionary
    test_set = Dataset(**test_dataset_cfg)

    print("regular", len(train_regular_set))
    print("anomaly", len(train_anomaly_set))

    train_regular_loader = DataLoader(
        train_regular_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator(device="cuda"),
    )

    train_anomaly_loader = DataLoader(
        train_anomaly_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True,
        generator=torch.Generator(device="cuda"),
    )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print("**************************************************************")
    print("regular", len(train_regular_set))
    print("anomaly", len(train_anomaly_set))
    print("test", len(test_set))
    if args.model_name == "mgfn":
        model = mgfn()
    elif args.model_name == "RF":
        model = RandomForestClassifier(n_estimators=1000, random_state=0)
    elif args.model_name == "RTFM":
        model = RTFM(args.feature_size, args.batch_size)
        print(
            """
        load RTFM
        """
        )
    elif args.model_name == "SVM":
        model = SVC(probability=True)
    else:
        model = S3R(args.feature_size, args.batch_size, args.quantize_size, dropout=args.dropout, modality=cfg.modality)

    # logger.info(train_regular_set.data_info)
    # logger.info(train_anomaly_set.data_info)
    # logger.info(test_set.data_info)
    # logger.info("Model Structure: \n{}".format(model))

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
                # score, score1 = inference3(test_loader, model, args, device)
                score = inference2(test_loader, train_regular_loader, train_anomaly_loader, model, args, device)
            elif args.model_name != "RF":
                score, cm = inference(test_loader, model, args, device)

            # >> Performance emsemble
            title = [["Dataset", "Method", "Feature", "AUC (%)"]]
            auc = [[args.dataset, args.model_name.upper(), args.backbone.upper(), f"{score*100.:.3f}"]]

            table = AsciiTable(title + auc, " Performance on {} ".format(args.dataset))
            for i in range(len(title[0])):
                table.justify_columns[i] = "center"
            logger.info("Summary Result on {} metric (emsemble)\n{}".format("AUC", table.table))
            print(cm)
            import seaborn as sns
            import matplotlib.pyplot as plt

            sns.heatmap(cm, annot=True, cmap="Blues")
            plt.savefig("sklearn_confusion_matrix_annot_blues.png")

        #             # >> Performance mean
        #             title = [['Dataset', 'Method', 'Feature', 'AUC (%)']]
        #             auc = [[args.dataset, args.model_name.upper(), args.backbone.upper(), f'{score1*100.:.3f}']]

        #             table = AsciiTable(title + auc, ' Performance on {} '.format(args.dataset))
        #             for i in range(len(title[0])):
        #                 table.justify_columns[i] = 'center'
        #             logger.info('Summary Result on {} metric (Mean)\n{}'.format('AUC', table.table))

        if args.inference:
            exit()
    else:
        if args.model_name not in ["RF", "SVM"]:
            score, cm = inference(test_loader, model, args, device)

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
    balance = [18911 // 32, 6614 // 32]

    if args.model_name not in ["RF", "SVM"]:
        for step in range(last_epoch, args.max_epoch + 1):
            if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr[step - 1]

            if (step - 1) % len(train_regular_loader) == 0:
                loadern_iter = iter(train_regular_loader)

            if (step - 1) % len(train_anomaly_loader) == 0:
                loadera_iter = iter(train_anomaly_loader)
            if args.model_name == "mgfn":
                loss = trainer(loadern_iter, loadera_iter, balance, model, args.batch_size, optimizer, device)
            else:
                loss = do_train(
                    loadern_iter, loadera_iter, balance, model, [args.batch_size] * 2, optimizer, device, args
                )

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
        with torch.set_grad_enabled(True):
            # print(len(train_regular_loader))
            for i in range(len(train_regular_loader)):
                if i % len(train_regular_loader) == 0:
                    loadern_iter = iter(train_regular_loader)
                if i % len(train_anomaly_loader) == 0:
                    loadera_iter = iter(train_anomaly_loader)
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
        filename = checkpoint_filename.format(data=args.dataset, model=args.model_name)
        model.fit(X_train, y_train)
        pickle.dump(model, open(args.checkpoint_path.joinpath(args.version).joinpath(filename), "wb"))


if __name__ == "__main__":
    main()
