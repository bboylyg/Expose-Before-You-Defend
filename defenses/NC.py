import argparse
import numpy as np
import sys
import json

import torch
from torch import Tensor, nn
import torchvision
import torchvision.transforms as transforms
import sys
from torch.utils.data import DataLoader
import time

sys.path.append('/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/shs/Multui-Trigger-Backdoor-Defense/')

from Multi_Trigger_Backdoor_Attack import *
import os
import matplotlib.pyplot as plt
import numpy as np
# from datasets.poison_tool_cifar import get_train_loader

def add_arguments(parser):
        parser.add_argument("--model_path", type=str, default = "/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/LfF-master(2)/checkpoint/cifar10/正常训练/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar")
        parser.add_argument('--data_parallel', type=str, default=True, help='Using torch.nn.DataParallel')
        parser.add_argument('--num_classes', type=int, default=10, help='number of classes') 
        parser.add_argument('--get_features', type=bool, default=False, help='whether get features as a part of output') 
        parser.add_argument('--input_height', type=int, default=32, help='height of the picture')
        parser.add_argument('--input_width', type=int, default=32, help='width of the picture')
        parser.add_argument('--input_channel', type=int, default=3, help='width of the picture')
        parser.add_argument("--n_times_test", type=int, default=1)
        parser.add_argument("--target_label", type=int)
        parser.add_argument("--bs", type=int, default=64)
        parser.add_argument("--lr", type=float, default=1e-1)
        parser.add_argument("--init_cost", type=float, default=1e-3)
        parser.add_argument("--atk_succ_threshold", type=float, default=98.0)
        parser.add_argument("--early_stop", type=bool, default=True)
        parser.add_argument("--early_stop_threshold", type=float, default=99.0)
        parser.add_argument("--early_stop_patience", type=int, default=25)
        parser.add_argument("--patience", type=int, default=5)
        parser.add_argument("--cost_multiplier", type=float, default=2)
        parser.add_argument("--epoch", type=int, default=20)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--EPSILON", type=float, default=1e-7)
        
parser = argparse.ArgumentParser(description=sys.argv[0])        
add_arguments(parser)
opt = parser.parse_args()



def create_dir(path_dir):
    list_subdir = path_dir.strip(".").split("/")
    list_subdir.remove("")
    base_dir = "./"
    for subdir in list_subdir:
        base_dir = os.path.join(base_dir, subdir)
        try:
            os.mkdir(base_dir)
        except:
            pass

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time  
term_width = int(85)
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()
    
class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone

def outlier_detection(l1_norm_list, idx_mapping, opt):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    median = torch.median(l1_norm_list)
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    if min_mad < 2:
        print("Not a backdoor model")
    else:
        print("This is a backdoor model")

#     if opt.to_file:
#         # result_path = os.path.join(opt.result, opt.saving_prefix, opt.dataset)
#         output_path = os.path.join(
#             result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset, opt.attack_mode)
#         )
#         with open(output_path, "a+") as f:
#             f.write(
#                 str(median.cpu().numpy()) + ", " + str(mad.cpu().numpy()) + ", " + str(min_mad.cpu().numpy()) + "\n"
#             )
#             l1_norm_list_to_save = [str(value) for value in l1_norm_list.cpu().numpy()]
#             f.write(", ".join(l1_norm_list_to_save) + "\n")

    flag_list = []
    for y_label in idx_mapping:
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:
        flag_list = sorted(flag_list, key=lambda x: x[1])

    print(
        "Flagged label list: {}".format(",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
    )
    
class RegressionModel(nn.Module):
    def __init__(self, opt, init_mask, init_pattern):
        self._EPSILON = opt.EPSILON
        super(RegressionModel, self).__init__()
        self.mask_tanh = nn.Parameter(torch.tensor(init_mask))
        self.pattern_tanh = nn.Parameter(torch.tensor(init_pattern))

        self.classifier = self._get_classifier(opt)
        self.normalizer = self._get_normalize(opt)
        self.denormalizer = self._get_denormalize(opt)

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
#         if self.normalizer:
#             pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self._EPSILON) + 0.5

#     def _get_classifier(self, opt):
#         if opt.dataset == "CIFAR10":
#             classifier = resnet18_cifar()
#         else:
#             raise Exception("Invalid Dataset")
#         # Load pretrained classifie
#         weight = '/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/LfF-master(2)/checkpoint/cifar10/正常训练/resnet18_Badnet_CIFAR10_epoch80_origacc:91.49_triglabelacc:100.tar'
#         state_dict = torch.load(weight)
#         classifier.load_state_dict(state_dict["state_dict"])
#         for param in classifier.parameters():
#             param.requires_grad = False
#         classifier.eval()
#         return classifier.to(device)
    
    def _get_classifier(self, opt):
        for name in ['CIFAR10', 'CIFAR100', 'ImageNette', 'SDog120Data', 'CUB200Data', 'Stanford40Data', 'MIT67Data', 'Flower102Data']:
            if name in opt.model_path:
                opt.dataset = name

        for name in ['EfficientNetB0', 'ResNet18', 'MobileNetV2', 'VGG16', 'ResNet34', 'ResNet50', 'PreActResNet18']:
            if name in opt.model_path:
                opt.model_name = name


        # 加载模型
        if any(name in opt.model_path for name in ['CIFAR10', 'CIFAR100']):
            model = select_model(opt, dataset=opt.dataset,
                                model_name=opt.model_name,
                                pretrained=True,
                                pretrained_models_path=opt.model_path,
                                )

        elif any(name in opt.model_path for name in ['ImageNette', 'SDog120Data', 'CUB200Data', 'Stanford40Data', 'MIT67Data', 'Flower102Data']):
            if 'ResNet18' in opt.model_path:
                import torchvision.models as models
                model = models.resnet18(pretrained=opt.pretrained, num_classes=opt.num_classes)
            elif opt.model_name == "vit_small_patch16_224":

                from model_for_cifar.vit import vit_small_patch16_224
                model = vit_small_patch16_224(pretrained=True, num_classes=opt.num_classes)

            # import timm
            # model = timm.create_model('vit_small_patch16_224',pretrained=True, num_classes=10)
            elif opt.model_name == "vit_base_patch16_224":
                from model_for_cifar.vit import vit_base_patch16_224
                model = vit_base_patch16_224(pretrained=True, num_classes=opt.num_classes)

            else:
                raise('model is not implemented')
                
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model

    def _get_denormalize(self, opt):
        if opt.dataset == "CIFAR10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "CIFAR10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        else:
            raise Exception("Invalid dataset")
        return normalizer


class Recorder:
    def __init__(self, opt):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = opt.init_cost
        self.cost_multiplier_up = opt.cost_multiplier
        self.cost_multiplier_down = opt.cost_multiplier ** 1.5

    def reset_state(self, opt):
        self.cost = opt.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):
#         result_dir = os.path.join(opt.result, opt.dataset)
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
#         result_dir = os.path.join(result_dir, opt.attack_mode)
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
#         result_dir = os.path.join(result_dir, str(opt.target_label))
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

#         path_mask = os.path.join(result_dir, "mask.png")
#         path_pattern = os.path.join(result_dir, "pattern.png")
#         path_trigger = os.path.join(result_dir, "trigger.png")

#         torchvision.utils.save_image(mask_best, path_mask, normalize=True)
#         torchvision.utils.save_image(pattern_best, path_pattern, normalize=True)
#         torchvision.utils.save_image(trigger, path_trigger, normalize=True)




class Neural_Cleanse():
    def __init__(self, init_mask, init_pattern):
        
        self.init_mask = init_mask
        self.init_pattern = init_pattern 
    
    def train_step(self, opt, regression_model, optimizerR, dataloader, recorder, epoch):
        print("Epoch {} - Label: {} | {} - {}:".format(epoch, opt.target_label, opt.dataset, opt.attack_type))
        # Set losses
        cross_entropy = nn.CrossEntropyLoss()
        total_pred = 0
        true_pred = 0

        # Record loss for all mini-batches
        loss_ce_list = []
        loss_reg_list = []
        loss_list = []
        loss_acc_list = []

        # Set inner early stop flag
        inner_early_stop_flag = False
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Forwarding and update model
            optimizerR.zero_grad()
            
            print(inputs.shape)
            
            inputs = inputs.to(device)
            sample_num = inputs.shape[0]
            total_pred += sample_num
            target_labels = torch.ones((sample_num), dtype=torch.int64).to(device) * self.opt.target_label
            predictions = regression_model(inputs)

            loss_ce = cross_entropy(predictions, target_labels)
            loss_reg = torch.norm(regression_model.get_raw_mask(), 1)
            total_loss = loss_ce + recorder.cost * loss_reg
            total_loss.backward()
            optimizerR.step()

            # Record minibatch information to list
            minibatch_accuracy = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
            loss_ce_list.append(loss_ce.detach())
            loss_reg_list.append(loss_reg.detach())
            loss_list.append(total_loss.detach())
            loss_acc_list.append(minibatch_accuracy)

            true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()
            progress_bar(batch_idx, len(dataloader))

        loss_ce_list = torch.stack(loss_ce_list)
        loss_reg_list = torch.stack(loss_reg_list)
        loss_list = torch.stack(loss_list)
        loss_acc_list = torch.stack(loss_acc_list)

        avg_loss_ce = torch.mean(loss_ce_list)
        avg_loss_reg = torch.mean(loss_reg_list)
        avg_loss = torch.mean(loss_list)
        avg_loss_acc = torch.mean(loss_acc_list)

        # Check to save best mask or not
        if avg_loss_acc >= opt.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()
            recorder.reg_best = avg_loss_reg
            recorder.save_result_to_dir(self.opt)
            print(" Updated !!!")

        # Show information
        print(
            "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
                true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
            )
        )

        # Check early stop
        if opt.early_stop:
            if recorder.reg_best < float("inf"):
                if recorder.reg_best >= self.opt.early_stop_threshold * recorder.early_stop_reg_best:
                    recorder.early_stop_counter += 1
                else:
                    recorder.early_stop_counter = 0

            recorder.early_stop_reg_best = min(recorder.early_stop_reg_best, recorder.reg_best)

            if (
                recorder.cost_down_flag
                and recorder.cost_up_flag
                and recorder.early_stop_counter >= self.opt.early_stop_patience
            ):
                print("Early_stop !!!")
                inner_early_stop_flag = True

        if not inner_early_stop_flag:
            # Check cost modification
            if recorder.cost == 0 and avg_loss_acc >= self.opt.atk_succ_threshold:
                recorder.cost_set_counter += 1
                if recorder.cost_set_counter >= self.opt.patience:
                    recorder.reset_state(self.opt)
            else:
                recorder.cost_set_counter = 0

            if avg_loss_acc >= opt.atk_succ_threshold:
                recorder.cost_up_counter += 1
                recorder.cost_down_counter = 0
            else:
                recorder.cost_up_counter = 0
                recorder.cost_down_counter += 1

            if recorder.cost_up_counter >= self.opt.patience:
                recorder.cost_up_counter = 0
                print("Up cost from {} to {}".format(recorder.cost, recorder.cost * recorder.cost_multiplier_up))
                recorder.cost *= recorder.cost_multiplier_up
                recorder.cost_up_flag = True

            elif recorder.cost_down_counter >= opt.patience:
                recorder.cost_down_counter = 0
                print("Down cost from {} to {}".format(recorder.cost, recorder.cost / recorder.cost_multiplier_down))
                recorder.cost /= recorder.cost_multiplier_down
                recorder.cost_down_flag = True

            # Save the final version
            if recorder.mask_best is None:
                recorder.mask_best = regression_model.get_raw_mask().detach()
                recorder.pattern_best = regression_model.get_raw_pattern().detach()

        return inner_early_stop_flag
    
    def train(self, opt):
        self.opt = opt

        from datasets.poison_tool_cifar import get_test_loader, get_train_loader  # 仿照RNP代码读取数据

        dataloader = get_train_loader(opt)

#         dataloader = get_train_dataloader()# 参数是？

        # 目前数据接口
        # clean_train = torchvision.datasets.CIFAR10(root='/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/LfF-master(2)/data/CIFAR10', train=True, download=True, transform=None)
        #
        # MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        # STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
        #
        # tf_test = torchvision.transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        #     ])
        #
        # _, split_set_clean_train = split_dataset(clean_train, frac=0.01)
        # clean_train_tf = DatasetTF(full_dataset=split_set_clean_train, transform=tf_test)
        # dataloader = DataLoader(clean_train_tf, batch_size=opt.bs, shuffle=True, num_workers=opt.num_workers)
        # 目前数据接口

        # Build regression model
        regression_model = RegressionModel(self.opt, self.init_mask, self.init_pattern).to(device)

        # Set optimizer
        optimizerR = torch.optim.Adam(regression_model.parameters(), lr=self.opt.lr, betas=(0.5, 0.9))

        # Set recorder (for recording best result)
        recorder = Recorder(self.opt)

        for epoch in range(self.opt.epoch):
            early_stop = self.train_step(self.opt, regression_model, optimizerR, dataloader, recorder, epoch)
            if early_stop:
                break


        # Save result to dir
        recorder.save_result_to_dir(self.opt)

        return recorder, self.opt

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if "all2one" in opt.model_path:
        opt.attack_type = 'multi_triggers_all2one'
    elif "all2all" in opt.model_path:
        opt.attack_type = 'multi_triggers_all2all'
    elif "all2rand" in opt.model_path:
        opt.attack_type = 'multi_triggers_all2rand'

    init_mask = np.ones((1, opt.input_height, opt.input_width)).astype(np.float32)
    init_pattern = np.ones((opt.input_channel, opt.input_height, opt.input_width)).astype(np.float32)
    
    NC = Neural_Cleanse(init_mask, init_pattern)

    for test in range(opt.n_times_test):
        print("Test {}:".format(test))

        masks = []
        idx_mapping = {}

        for target_label in range(opt.num_classes):
            print("----------------- Analyzing label: {} -----------------".format(target_label))
            opt.target_label = target_label
            print(opt.target_label)
            recorder, opt = NC.train(opt)

            mask = recorder.mask_best
            masks.append(mask)
            idx_mapping[target_label] = len(masks) - 1
            
            torch.cuda.empty_cache()

        l1_norm_list = torch.stack([torch.norm(m, p=1) for m in masks])
        print("{} labels found".format(len(l1_norm_list)))
        print("Norm values: {}".format(l1_norm_list))
        outlier_detection(l1_norm_list, idx_mapping, opt)
