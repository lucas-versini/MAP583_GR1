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
from dataset import FaceDataset, FaceDatasetMultiTask
from defaults import _C as cfg

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import matplotlib.pyplot as plt

name_variables = ["gender", "race", "makeup", "time", "happiness"]
class_variables = [2, 3, 4, 2, 4] # Number of classes for each variable
index_variables = np.cumsum([101] + class_variables)[:-1] # Index of the first class for each variable

dic_gender = {'male': 0, 'female': 1}
dic_race = {'caucasian': 0, 'asian': 1, 'afroamerican': 2}
dic_makeup = {'verysubtle': 0, 'makeup': 1, 'nomakeup': 2, 'notclear': 3}
dic_time = {'modernphoto': 0, 'oldphoto': 1}
dic_happiness = {'slightlyhappy': 0, 'neutral': 1, 'happy': 2, 'other': 3}

# Fonctions pour transformer les variables en entiers
def f_gender(x):
    return dic_gender[x]
def f_race(x):
    return dic_race[x]
def f_makeup(x):
    return dic_makeup[x]
def f_time(x):
    return dic_time[x]
def f_happiness(x):
    return dic_happiness[x]
variables_to_int = [f_gender, f_race, f_makeup, f_time, f_happiness]

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


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    loss_monitor = AverageMeter()
    list_accuracy_monitor = [AverageMeter() for _ in range(len(name_variables))]
    accuracy_monitor_age = AverageMeter()

    with tqdm(train_loader) as _tqdm:
        for x, (y_age, y) in _tqdm:
            x, y_age, y = x.to(device), y_age.to(device), y.to(device)

            # compute output
            outputs = model(x)

            # calc loss
            loss = criterion(outputs[:, :101], y_age) + sum([criterion(outputs[:, ind:ind+leng], y[:, i]) for i, (ind, leng) in enumerate(zip(index_variables, class_variables))])
            
            cur_loss = loss.item()

            # calc accuracy

                # Variables other than age
            list_correct_num = []
            for i, (ind, leng) in enumerate(zip(index_variables, class_variables)):
                _, predicted = outputs[:, ind:ind+leng].max(1)
                correct_num = predicted.eq(y[:, i]).sum().item()
                list_correct_num.append(correct_num)
                list_accuracy_monitor[i].update(correct_num, x.size(0))

                # Age
            _, predicted = outputs[:, :101].max(1)
            correct_num_age = predicted.eq(y_age).sum().item()
            accuracy_monitor_age.update(correct_num_age, x.size(0))

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                              acc=accuracy_monitor_age.avg, correct=correct_num_age, sample_num=sample_num)

    return loss_monitor.avg, [x.avg for x in list_accuracy_monitor], accuracy_monitor_age.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()

    list_accuracy_monitor = [AverageMeter() for _ in range(len(name_variables))]
    list_age_accuracy_monitor = [AverageMeter() for _ in [0, 1, 2, 5, -1]]

    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, (y_age, y) in _tqdm:
                x, y_age, y = x.to(device), y_age.to(device), y.to(device)

                # compute output
                outputs = model(x)
                gt.append(y_age.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs[:, :101], y_age) + sum([criterion(outputs[:, ind:ind+leng], y[:, i]) for i, (ind, leng) in enumerate(zip(index_variables, class_variables))]) # Softmax + cross entropy
                    cur_loss = loss.item()

                    # calc accuracy
                    for i in range(len(name_variables)):
                        _, predicted = outputs[:, index_variables[i]:index_variables[i]+class_variables[i]].max(1)
                        correct_num = predicted.eq(y[:, i]).sum().item()
                        list_accuracy_monitor[i].update(correct_num, x.size(0))
                    # Age
                    preds.append(F.softmax(outputs[:, :101], dim=-1).cpu().numpy())
                    predicted = (F.softmax(outputs[:, :101], dim=-1) * torch.arange(0, 101, dtype=outputs.dtype, device=outputs.device)).sum(dim=1)
                    predicted = predicted.round().clip(0, 100).long()
                    correct_num = predicted.eq(y_age).sum().item()
                    list_age_accuracy_monitor[0].update(correct_num, x.size(0))
                    correct_num_1 = (predicted - y_age).abs().le(1).sum().item()
                    list_age_accuracy_monitor[1].update(correct_num_1, x.size(0))
                    correct_num_2 = (predicted - y_age).abs().le(2).sum().item()
                    list_age_accuracy_monitor[2].update(correct_num_2, x.size(0))
                    correct_num_5 = (predicted - y_age).abs().le(5).sum().item()
                    list_age_accuracy_monitor[3].update(correct_num_5, x.size(0))
                    _, predicted = outputs[:, :101].max(1)
                    correct_num_max = predicted.eq(y_age).sum().item()
                    list_age_accuracy_monitor[4].update(correct_num_max, x.size(0))

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                                      acc=list_age_accuracy_monitor[-1].avg, correct=correct_num_max, sample_num=sample_num)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, [x.avg for x in list_accuracy_monitor], [x.avg for x in list_age_accuracy_monitor], mae, ave_preds, gt


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
    model = get_model(model_name=cfg.MODEL.ARCH) # None sinon ne marche pas... Non, voir fin des imports

    n_classes = sum(class_variables) + 101
    model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)

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
    train_dataset = FaceDatasetMultiTask(args.data_dir, "train", img_size=cfg.MODEL.IMG_SIZE, augment=True,
                                age_stddev=cfg.TRAIN.AGE_STDDEV, name_variables=name_variables, variables_to_int=variables_to_int)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS, drop_last=True)

    val_dataset = FaceDatasetMultiTask(args.data_dir, "valid", img_size=cfg.MODEL.IMG_SIZE, augment=False, name_variables=name_variables, variables_to_int=variables_to_int)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS, drop_last=False)

    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY_RATE,
                       last_epoch=start_epoch - 1)
    best_val_mae = 10000.0
    train_writer = None

    list_train_loss = []
    list_train_acc = [ [] for _ in range(len(name_variables)) ]
    list_train_acc_age = []

    list_val_loss = []
    list_val_acc = [ [] for _ in range(len(name_variables)) ]
    list_val_acc_age = [ [] for _ in range(5)]

    list_val_mae = []

    list_models_saved = []

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents = True, exist_ok = True)
    with open(graphs_dir.joinpath('values.txt'), 'a') as f:
        f.write("Not age: " + str(name_variables) + "\n")
        f.write("Age: 0, 1, 2, 5, max\n")

    print(f"Device = {device}")
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        print(f"Epoch {epoch} / {cfg.TRAIN.EPOCHS}")
        # train
        train_loss, list_accuracy_monitor_train, accuracy_age = train(train_loader, model, criterion, optimizer, epoch, device)
        
        # validate
        val_loss, list_accuracy_monitor_val, list_age_accuracy_monitor, val_mae, ave_preds, gt = validate(val_loader, model, criterion, epoch, device)
        
        #######
        list_train_loss.append(train_loss)
        list_train_acc = [list_train_acc[i] + [list_accuracy_monitor_train[i]] for i in range(len(name_variables))]
        list_train_acc_age.append(accuracy_age)

        list_val_acc = [list_val_acc[i] + [list_accuracy_monitor_val[i]] for i in range(len(name_variables))]
        list_val_acc_age = [list_val_acc_age[i] + [list_age_accuracy_monitor[i]] for i in range(len(list_age_accuracy_monitor))]
        list_val_mae.append(val_mae)

        # Save values
        with open(graphs_dir.joinpath('values.txt'), 'a') as f:
            f.write(str(epoch) + "\n")
            f.write("Train_loss " + str(train_loss) + "\n")
            f.write("Train_acc (no age)" + str(list_accuracy_monitor_train) + "\n")
            f.write("Train_acc (age) " + str(accuracy_age) + "\n")

            f.write("Val_loss " + str(val_loss) + "\n")
            f.write("Val_acc (no age) " + str(list_val_acc) + "\n")
            f.write("Val_acc (age) " + str(list_accuracy_monitor_val) + "\n")
            f.write("Val_mae " + str(val_mae) + "\n")
            f.write("\n")

        # Plot and save the graphs
            # Loss
        plt.figure()
        plt.plot(list_train_loss, label='train_loss')
        plt.plot(list_val_loss, label='val_loss')
        plt.legend()
        plt.title('Loss, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('loss.png'))
        plt.close()

            # Accuracies (all but age)
        for i in range(len(name_variables)):
            plt.figure()
            plt.plot(list_train_acc[i], label='train_acc')
            plt.plot(list_val_acc[i], label='val_acc')
            plt.legend()
            plt.title(f'Accuracy for {name_variables[i]}, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
            plt.savefig(graphs_dir.joinpath(f'accuracy_{name_variables[i]}.png'))
            plt.close()

            # Accuracies (age)
        labels = ["0", "1", "2", "5", "max"]
        plt.figure()
        plt.plot(list_train_acc_age, label='train_acc')
        for i in range(len(list_age_accuracy_monitor)):
            plt.plot(list_val_acc_age[i], label=f'val_acc_{labels[i]}')
        plt.legend()
        plt.title(f'Accuracy for the age, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('accuracy_age.png'))
        plt.close()

            # MAE
        plt.figure()
        plt.plot(list_val_mae, label='val_mae')
        plt.legend()
        plt.title('Mean Absolute Error, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath('mae.png'))
        plt.close()

            # Prediction vs ground truth
        plt.figure()
        plt.plot(gt, ave_preds, 'ro', label = "Prediction")
        plt.plot(gt, gt, '-b', label = "Ground truth")
        plt.legend()
        plt.title('Prediction vs ground truth, epoch ' + str(epoch) + ' / ' + str(cfg.TRAIN.EPOCHS) + ' epochs')
        plt.savefig(graphs_dir.joinpath(f'gt_vs_res.png'))
        plt.close()

            # Save the values
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
