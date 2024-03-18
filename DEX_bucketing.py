import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import FaceDataset
from defaults import _C as cfg

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt
import os

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    
    parser.add_argument("--graphs_dir", type=str, default="graphs", help = "Graphs directory")

    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device, age_buckets):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            # compute output
            outputs = model(x)

            ## update y
            y = age_buckets(y)

            # calc loss
            loss = criterion(outputs, y)

            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    return loss_monitor.avg, accuracy_monitor.avg

def validate(validate_loader, model, criterion, epoch, device, validate_dataset, age_buckets, bucket_to_mean):
    model.eval()
    loss_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for _, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                gt.append(y.cpu().numpy())

                ## update y
                y = age_buckets(y)

                # compute output
                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)

                    # bucket centers tensor
                    centers = torch.tensor(bucket_to_mean, dtype=outputs.dtype, device=outputs.device)
                    predicted = (F.softmax(outputs, dim=-1) * centers).sum(dim=1)
                    predicted = predicted.round().clip(0, 100).long()
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg), correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    age_centers = np.array(bucket_to_mean)
    avg_preds_class = preds.argmax(axis=-1)
    avg_preds_class = bucket_to_mean[avg_preds_class]
    avg_preds_DEX = (preds * age_centers).sum(axis=-1)
    diff = avg_preds_class - gt
    diff_DEX = avg_preds_DEX - gt
    mae = np.abs(diff).mean()
    mae_DEX = np.abs(diff_DEX).mean()
    epsilon_error = 1 - np.exp(-diff**2/2/np.array(validate_dataset.std)**2).mean()
    epsilon_error_DEX = 1 - np.exp(-diff_DEX**2/2/np.array(validate_dataset.std)**2).mean()

    return loss_monitor.avg, avg_preds_class, avg_preds_DEX, gt, mae, mae_DEX, epsilon_error, epsilon_error_DEX


def train_and_test(out_neurons=101, bucketing_method='equal'):
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    #
    print("Output neurons: ", out_neurons)
    print("Bucketing method: ", bucketing_method)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    print("model name: ", cfg.MODEL.ARCH)

    ## Here, change num_classes from base 101
    n_classes = out_neurons
    model = get_model(model_name=cfg.MODEL.ARCH, num_classes=n_classes)

    if cfg.TRAIN.OPT == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR,
                                    momentum=cfg.TRAIN.MOMENTUM,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDataset(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDataset(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0

    # Define array containing all ages
    ages = np.array([item[1] for item in train_dataset])
    ages = np.sort(ages)

    ## Define the buckets for the age. Also define bucket_to_mean, which gives the mean age for each bucket
    if bucketing_method == 'equal':
        age_buckets = lambda x: (x / 101 * n_classes).long().clip(0, n_classes - 1)
        bucket_to_mean = np.linspace(0, 100, n_classes, endpoint=False) + 100 / n_classes / 2
    elif bucketing_method == 'balanced':
        buckets_array = np.percentile(ages, np.linspace(0, 100, n_classes, endpoint=False))
        buckets = torch.from_numpy(buckets_array).to(device)
        age_buckets = lambda x: (torch.bucketize(x, buckets) - 1).clip(0, n_classes - 1)
        bucket_to_mean = np.array([])
        for i in range(n_classes - 1):
            sliced = ages[(ages >= buckets_array[i]) & (ages < buckets_array[i+1])]
            if len(sliced) == 0:
                bucket_to_mean = np.append(bucket_to_mean, (buckets_array[i] + buckets_array[i+1]) / 2)
            else:
                bucket_to_mean = np.append(bucket_to_mean, np.mean(sliced))
        sliced = ages[ages >= buckets_array[-1]]
        if len(sliced) == 0:
            bucket_to_mean = np.append(bucket_to_mean, buckets_array[-1])
        bucket_to_mean = np.append(bucket_to_mean, np.mean(sliced))
    else:
        raise ValueError("Invalid bucketing method")
    print("Bucketed using ", bucketing_method)

    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_ave_preds = []
    list_ave_preds_DEX = []
    list_gt = []
    list_mae = []
    list_mae_DEX = []
    list_epsilon_error = []
    list_epsilon_error_DEX = []

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents = True, exist_ok = True)

    print(f"Device = {device}")


    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        print(f"Epoch {epoch} / {cfg.TRAIN.EPOCHS}")
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, age_buckets)

        # validate
        val_loss, ave_preds, ave_preds_DEX, gt, mae, mae_DEX, epsilon_error, epsilon_error_DEX = validate(val_loader, model, criterion, epoch, device, val_dataset, age_buckets, bucket_to_mean)

        #######
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_val_loss.append(val_loss)
        list_ave_preds.append(ave_preds)
        list_ave_preds_DEX.append(ave_preds_DEX)
        list_gt.append(gt)
        list_mae.append(mae)
        list_mae_DEX.append(mae_DEX)
        list_epsilon_error.append(epsilon_error)
        list_epsilon_error_DEX.append(epsilon_error_DEX)

        list_models_saved = []

        #######

        # checkpoint
        print(f"Device: {device}")
        val_mae = mae_DEX
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")

    return list_ave_preds, list_ave_preds_DEX, list_gt, list_mae, list_mae_DEX, list_epsilon_error, list_epsilon_error_DEX

if __name__ == '__main__':
    for out_neurons in [10,25,37,50,100]:
        for method in ['equal', 'balanced']:
            list_ave_preds, list_ave_preds_DEX, list_gt, list_mae, list_mae_DEX, list_epsilon_error, list_epsilon_error_DEX = train_and_test(out_neurons, method)
            
            # Output the results in values.txt, in graphs_dir
            graphs_dir = Path("graphs")
            graphs_dir.mkdir(parents = True, exist_ok = True)

            with open(graphs_dir.joinpath("values.txt"), "a") as f:
                f.write(f"{out_neurons}, {method}\n")
                #f.write(f"{','.join(map(str, list_ave_preds))}\n")
                #f.write(f"{','.join(map(str, list_ave_preds_DEX))}\n")
                #f.write(f"{','.join(map(str, list_gt))}\n")
                f.write(f"{','.join(map(str, list_mae))}\n")
                f.write(f"{','.join(map(str, list_mae_DEX))}\n")
                f.write(f"{','.join(map(str, list_epsilon_error))}\n")
                f.write(f"{','.join(map(str, list_epsilon_error_DEX))}\n")
                f.write("\n")