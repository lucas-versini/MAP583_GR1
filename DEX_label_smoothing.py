import argparse
import better_exceptions
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

def vectorized_gaussian_label_smoothing(labels, num_classes = 101, sigma=1.6e-2, smoothing=0.2):
    """
    Apply Gaussian label smoothing in a vectorized manner.
    
    :param labels: Batch of labels, shape [batch_size]
    :param num_classes: Total number of classes
    :param sigma: Standard deviation for the Gaussian distribution
    :param smoothing: Smoothing factor
    :return: Smoothed labels, shape [batch_size, num_classes]
    """
    # Generate a grid of class indices [num_classes, num_classes]
    indices = torch.arange(0, num_classes).unsqueeze(0).to(labels.device)
    # Expand labels to match the shape of indices
    labels_ = labels.unsqueeze(1).expand(-1, num_classes)
    
    # Calculate the Gaussian distribution
    gaussian_dist = torch.exp(-0.5 * (((indices - labels_.float()) / num_classes) ** 2) / sigma ** 2)
    # print(gaussian_dist[10, max(labels[10].item()-5, 0):(labels[10].item()+6)])
    # 0 for the true label in the gaussian distribution
    gaussian_dist[indices == labels_] = 0
    gaussian_dist /= gaussian_dist.sum(1, keepdim=True)  # Normalize
    # print(gaussian_dist[10, max(labels[10].item()-5, 0):(labels[10].item()+6)])
    
    # Mix the Gaussian distribution with the smoothing
    one_hot = torch.nn.functional.one_hot(labels, num_classes).float()
    smoothed_labels = (1 - smoothing) * one_hot + smoothing * gaussian_dist

    gt = labels[10].item()
    # print(smoothed_labels[10, max(gt-2, 0):(gt+3)])
    
    return smoothed_labels

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, y_hard in _tqdm:
            x = x.to(device)
            y_hard = y_hard.to(device)
            y = vectorized_gaussian_label_smoothing(y_hard)

            # compute output
            outputs = model(x)

            # calc loss
            loss = criterion(outputs, y)

            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y_hard).sum().item()

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


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    accuracy_monitor_1 = AverageMeter()
    accuracy_monitor_2 = AverageMeter()
    accuracy_monitor_5 = AverageMeter()
    accuracy_monitor_max = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num_max = predicted.eq(y).sum().item()

                    predicted = (F.softmax(outputs, dim=-1) * torch.arange(0, 101, dtype=outputs.dtype, device=outputs.device)).sum(dim=1)
                    predicted = predicted.round().clip(0, 100).long()
                    correct_num = predicted.eq(y).sum().item()
                    correct_num_1 = (predicted - y).abs().le(1).sum().item()
                    correct_num_2 = (predicted - y).abs().le(2).sum().item()
                    correct_num_5 = (predicted - y).abs().le(5).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    accuracy_monitor_1.update(correct_num_1, sample_num)
                    accuracy_monitor_2.update(correct_num_2, sample_num)
                    accuracy_monitor_5.update(correct_num_5, sample_num)
                    accuracy_monitor_max.update(correct_num_max, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, accuracy_monitor_1.avg, accuracy_monitor_2.avg, accuracy_monitor_5.avg, accuracy_monitor_max.avg, mae, ave_preds, gt


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    print("model name: ", cfg.MODEL.ARCH)
    model = get_model(model_name=cfg.MODEL.ARCH)

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

    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []
    list_val_acc_1 = []
    list_val_acc_2 = []
    list_val_acc_5 = []
    list_val_acc_max = []
    list_val_mae = []
    list_models_saved = []

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents = True, exist_ok = True)

    print(f"Device = {device}")
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        print(f"Epoch {epoch} / {cfg.TRAIN.EPOCHS}")
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_acc_1, val_acc_2, val_acc_5, val_acc_max, val_mae, ave_preds, gt = validate(val_loader, model, criterion, epoch, device)

        #######
        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)
        list_val_acc_1.append(val_acc_1)
        list_val_acc_2.append(val_acc_2)
        list_val_acc_5.append(val_acc_5)
        list_val_acc_max.append(val_acc_max)
        list_val_mae.append(val_mae)

        with open(graphs_dir.joinpath('values.txt'), 'a') as f:
            f.write(str(epoch) + "\n")
            f.write("Train_loss " + str(train_loss) + "\n")
            f.write("Train_acc " + str(train_acc) + "\n")
            f.write("Val_loss " + str(val_loss) + "\n")
            f.write("Val_acc " + str(val_acc) + "\n")
            f.write("Val_acc_1 " + str(val_acc_1) + "\n")
            f.write("Val_acc_2 " + str(val_acc_2) + "\n")
            f.write("Val_acc_5 " + str(val_acc_5) + "\n")
            f.write("Val_acc_max " + str(val_acc_max) + "\n")
            f.write("Val_mae " + str(val_mae) + "\n")
            f.write("\n")

        # Plot and save the graph
        plt.figure()
        plt.plot(list_train_loss, label = 'train_loss')
        plt.plot(list_val_loss, label='val_loss')
        plt.legend()
        plt.title('Loss, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('loss.png'))
        plt.close()

        plt.figure()
        plt.plot(list_train_acc, label='train_acc')
        plt.plot(list_val_acc, label='val_acc')
        plt.plot(list_val_acc_1, label='val_acc_1')
        plt.plot(list_val_acc_2, label='val_acc_2')
        plt.plot(list_val_acc_5, label='val_acc_5')
        plt.plot(list_val_acc_max, label='val_acc_max')
        plt.legend()
        plt.title('Accuracy, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('accuracy.png'))
        plt.close()

        plt.figure()
        plt.plot(list_val_mae, label='val_mae')
        plt.legend()
        plt.title('Mean Absolute Error, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('mae.png'))
        plt.close()

        plt.figure()
        plt.plot(gt, ave_preds, 'ro', label = "Prediction")
        plt.plot(gt, gt, '-b', label = "Ground truth")
        plt.legend()
        plt.title('Prediction vs ground truth, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath(f'gt_vs_res.png'))
        plt.close()

        with open(graphs_dir.joinpath('gt_vs_res.txt'), 'w') as f:
            f.write("Ground truth | result\n")
            for i in range(len(gt)):
                f.write(str(gt[i]) + " | " + str(ave_preds[i]) + "\n")

        #######

        # checkpoint
        print(f"Device: {device}")
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            
            for model_saved in list_models_saved:
                # Delete
                Path(checkpoint_dir.joinpath(model_saved)).unlink()
                print(f"Deleted {model_saved}")
            list_models_saved = []

            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': cfg.MODEL.ARCH,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            list_models_saved.append("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae))
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step()

    print("=> training finished")
    print(f"additional opts: {args.opts}")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()
