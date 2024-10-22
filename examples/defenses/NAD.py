# from train import eval
import argparse
import logging
from taibackdoor.datasets.poison_tool_cifar import get_test_loader, get_train_loader
from taibackdoor.defenses.NAD import NAD

# 日志
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
# backdoor attacks
parser.add_argument('--cuda', type=int, default=1, help='cuda available')
parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
parser.add_argument('--backdoor_model_path', type=str,
                    default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
                    help='path of backdoored model')
parser.add_argument('--arch', type=str, default='resnet18',
                    choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                             'vgg19_bn'])
parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')

# backdoor attacks
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')

# NAD
parser.add_argument("--lambda_at", type=int, default = 5000, help='value of lambda')
parser.add_argument("--lr", type=float, default = 0.01, help='learning rate')
parser.add_argument("--ratio", type=float, default = 0.01, help='ratio of clean data')
parser.add_argument("--print_freq", type=int, default = 400)
parser.add_argument("--model_path", type=str, default = '/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/LfF-master(2)/checkpoint/cifar10/正常训练/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar', help='path of backdoored model')
parser.add_argument("--epochs", type=int, default = 10, help='epoch of distillation')

args = parser.parse_args()

args.clean_train_loader = get_train_loader(args)
args.test_clean_loader, args.test_bad_loader = get_test_loader(args)
    

NAD = NAD(args, logger)
NAD.AT()