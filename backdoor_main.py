import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from collections import OrderedDict
import models
from models.model_for_cifar.model_select import select_model
from datasets.poison_tool_cifar import get_backdoor_loader, get_test_loader

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

seed = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)


def train_step(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        loss.backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def save_checkpoint(state, epoch, is_best, args):
    if is_best:
        filepath = os.path.join(args.save_root, args.model_name + '_' + args.trigger_type + '_' + args.dataset + '_' + f'target_label{args.target_label}' + '_' + f'poison_rate{args.poison_rate}' + '_' + f'epoch{epoch}.tar')
        torch.save(state, filepath)

def main(args):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.log_root, 'output.log')),
            logging.StreamHandler()
        ])
    logger.info(args)

    logger.info('----------- Backdoored Data Initialization --------------')
    _, backdoor_data_loader = get_backdoor_loader(args)
    clean_test_loader, bad_test_loader = get_test_loader(args)

    logger.info('----------- Backdoor Model Initialization --------------')
    net = select_model(args, dataset=args.dataset,
                            model_name=args.model_name,
                            pretrained=False,
                            pretrained_models_path=None
                            )
    print(net)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    logger.info('----------- Backdoor Model Training--------------')
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(0, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(args=args, model=net, criterion=criterion, optimizer=optimizer,
                                      data_loader=backdoor_data_loader)
        cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
        po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=bad_test_loader)
        scheduler.step()
        end = time.time()
        logger.info(
            '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
            cl_test_loss, cl_test_acc)


        if epoch % args.save_every == 0 and epoch != 0 and epoch >= 50:
            # save checkpoint at interval epoch
            is_best = True
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'clean_acc': cl_test_acc,
                'bad_acc': po_test_acc,
                'optimizer': optimizer.state_dict(),
            }, epoch, is_best, args)
            
            logger.info('[INFO] Save model weight epoch {}'.format(epoch))

def get_arguments():
    parser = argparse.ArgumentParser()

   # various path
    parser.add_argument('--cuda', type=int, default=1, help='cuda available')
    parser.add_argument('--save_every', type=int, default=5, help='save checkpoints every few epochs')
    parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
    parser.add_argument('--log_name', type=str, default=None, help='logs name')
    parser.add_argument('--save', type=int, default=1, help='whether save the weight')
    parser.add_argument('--save_root', type=str, default='weights/', help='where to save the weight')
    parser.add_argument('--model_name', type=str, default='ResNet18',
                        choices=['ResNet18', 'vit_small_patch16_224'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[40, 80],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--lr', type=int, default=0.1, help='the number of epochs for unlearning')
    parser.add_argument('--epochs', type=int, default=60, help='the number of epochs for training')

    # VITs CIFAR10
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=32)


    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
    parser.add_argument('--poison_rate', type=float, default=0.1, help='ratio of backdoor poisoned data')

    return parser

if __name__ == '__main__':
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    args = get_arguments().parse_args()


    if args.dataset == 'CIFAR10':
        model_names = ['ResNet18']

        # trigger_pools_cifar = ['onePixelTrigger', 'gridTrigger', 'wanetTrigger', 'trojanTrigger', 'blendTrigger',
        #                      'signalTrigger', 'CLTrigger', 'smoothTrigger', 'dynamicTrigger', 'nashTrigger']

        trigger_pools_cifar = ['onePixelTrigger']       
        poison_rates = [0.0]
        # args.epochs = 1

        for model_name in model_names:
            args.model_name = model_name
            # args.trigger_types = trigger_pools_cifar
            # print(args.model_name)
            if args.model_name == 'vit_small_patch16_224' or args.model_name == 'vit_base_patch16_224':
                args.lr = 0.001
                args.epochs = 11
                args.img_size = 224
                args.schedule = [10, 20]
            for trigger_type in trigger_pools_cifar:
                args.trigger_type = trigger_type
                # print('attack_type:', args.attack_type) 
                for poison_rate in poison_rates:
                    args.poison_rate = poison_rate
                    main(args)