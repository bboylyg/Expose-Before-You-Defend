import argparse
import mlconfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import taibackdoor
import taibackdoor.models
import taibackdoor.datasets
import os
import sys
import numpy as np
from taibackdoor.training.exp_mgmt import ExperimentManager
from taibackdoor.training import util
from torch.autograd import Variable

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='HiddenVision')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='cifar10_blend_rn18', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_best_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)


def save_model(is_best):
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    exp.save_state(scheduler, 'scheduler_state_dict')

    if is_best:
        exp.save_state(model, 'model_state_dict_best')
        exp.save_state(optimizer, 'optimizer_state_dict_best')
        exp.save_state(scheduler, 'scheduler_state_dict_best')

    return


def epoch_exp_stats():
    # Set epoch level experiment tracking
    # Track Training Loss
    stats = {}
    model.eval()
    tracker_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    train_loss_list, correct_list = [], []
    for images, labels in no_shuffle_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        logits = model(images)
        loss = tracker_criterion(logits, labels)
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels)
        train_loss_list += loss.detach().cpu().numpy().tolist()
        correct_list += correct.detach().cpu().numpy().tolist()

    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


def eval(target_model, epoch, loader):
    target_model.eval()
    # Training Evaluations
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    loss_list, correct_list = [], []
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        batch_size = images.shape[0]
        with torch.no_grad():
            logits = target_model(images)
            loss = F.cross_entropy(logits, labels, reduction='none')

        loss_list += loss.detach().cpu().numpy().tolist()
        loss = loss.mean().item()
        # Calculate acc
        acc = util.accuracy(logits, labels, topk=(1,))[0].item()
        # Update Meters
        loss_meters.update(loss, batch_size)
        acc_meters.update(acc, batch_size)

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels)
        correct_list += correct.detach().cpu().numpy().tolist()

    return loss_meters.avg, acc_meters.avg, loss_list, correct_list