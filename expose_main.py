import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import argparse
import os

from datasets.poison_tool_cifar import get_test_loader, get_train_loader, split_dataset
from exposes import unlearn, sfft, awp, act_prune
from exposes.utils import load_state_dict
import models
from models.model_for_cifar.resnet_cifar import NoisyBatchNorm2d

import sys

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# seed = 98
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(seed)
# np.random.seed(seed)

sys.path.append('/data/gpfs/projects/punim0619/yige/taibackdoor')
# os.chdir('/data/gpfs/projects/punim0619/yige/taibackdoor')


if __name__ == '__main__':  
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler('output.log'),
            logging.StreamHandler()
        ])

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='type of dataset')
    parser.add_argument('--ratio', type=int, default=0.01, help='ratio of defense data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)

    parser.add_argument('--backdoor_model_path', type=str,
                        default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
                        help='path of backdoored model')
    parser.add_argument('--output_model_path', type=str,
                        default=None, help='path of unlearned backdoored model')
    parser.add_argument('--output_logs_path', type=str,
                        default='exposes/logs/', help='path of logger')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])

    args = parser.parse_args()

    _, split_set = split_dataset(dataset_name=args.dataset, ratio=args.ratio, perm=None)
    defense_data_loader = DataLoader(split_set, batch_size=128, shuffle=True, num_workers=4)
    clean_test_loader, bad_test_loader = get_test_loader(args)



    logger.info('----------- Data Initialization --------------')
    data_loader = {'defense_loader': defense_data_loader,
                   'clean_test_loader': clean_test_loader,
                   'bad_test_loader': bad_test_loader
                   }

    logger.info('----------- Model Initialization --------------')

    state_dict = torch.load(args.backdoor_model_path, map_location=device)
    net = getattr(models.model_for_cifar.resnet_cifar, args.arch)(num_classes=args.num_classes, norm_layer=None)
    load_state_dict(net, orig_state_dict=state_dict)
    net = net.to(device)

    logger.info('----------- Model Exposing Strategy --------------')

    unlearn = unlearn.Unlearning(args, logger, net, data_loader)
    unlearn.do_expose()

    # shuffleFT = sfft.ShuffleFT(args, logger, net, data_loader)
    # shuffleFT.do_expose()

    # actPruning = act_prune.ActPruning(args, logger, net, data_loader)
    # actPruning.do_expose()

    # mask_net = getattr(models.model_for_cifar.resnet_cifar, args.arch)(num_classes=args.num_classes, norm_layer=NoisyBatchNorm2d)
    # load_state_dict(mask_net, orig_state_dict=state_dict)
    # mask_net = mask_net.to(device)
    # awp = awp.AWP(args, logger, mask_net, data_loader)
    # awp.do_expose()

