from train import eval
import argparse
import logging
from taibackdoor.datasets.poison_tool_cifar import get_test_loader, get_train_loader
from taibackdoor.defenses.ANP import anp


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

# 参数设定
parser = argparse.ArgumentParser()
parser.add_argument('--target_label', type=int, default=0, help='class of target label')
parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--model_path', type=str, default='model_last.th')
args = parser.parse_args()

# 数据集
defense_data_loader = get_train_loader(args)
clean_test_loader, bad_test_loader = get_test_loader(args)

args.defense_data_loader = defense_data_loader
args.clean_test_loader = clean_test_loader
args.bad_test_loader = bad_test_loader
args.eval = eval #验证函数

# ANP防御
ANP = anp(args, logger)
ANP.anp_mask()
ANP.prune_neuron()
