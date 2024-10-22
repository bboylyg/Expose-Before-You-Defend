
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Optimizer, lr_scheduler
from models.model_for_cifar.resnet_cifar import NoisyBatchNorm2d

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class AWP():
    def __init__(self, args, logger, mask_net, data_loader, **kwargs) -> None:
        self.args = args
        self.arguments()
        self.logger = logger

        self.logger.info(self.args)
        self.mask_net = mask_net
        self.defense_loader = data_loader['defense_loader']
        self.clean_test_loader = data_loader['clean_test_loader']
        self.bad_test_loader = data_loader['bad_test_loader']
    
    def arguments(self):
        if 'discription' not in self.args:
            self.args.discription = 'AWP'
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'print_every' not in self.args:
            self.args.print_every = 500
        if 'model_path' not in self.args:
            self.args.model_path = 'model_last.th'
        if 'num_epochs' not in self.args:
            self.args.num_epochs = 5
        if 'anp_eps' not in self.args:
            self.args.anp_eps = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if 'anp_steps' not in self.args:
            self.args.anp_steps = 1
        if 'anp_alpha' not in self.args:
            self.args.anp_alpha = 0.2
        if 'mask_optim_lr' not in self.args:
            self.args.mask_optim_lr = 0.2
        if 'noise_optim_lr' not in self.args:
            self.args.noise_optim_lr = 0.2
        if 'rand_init' not in self.args:
            self.args.rand_init = True
        if 'pruning_max' not in self.args:
            self.args.pruning_max = 0.95
        if 'pruning_step' not in self.args:
            self.args.pruning_step = 0.05
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def reset(mask_net: nn.Module, rand_init: bool, anp_eps: float) -> None:
        for m in mask_net.modules():
            if isinstance(m, NoisyBatchNorm2d):
                m.reset(rand_init=rand_init, eps=anp_eps)

    @staticmethod
    def sign_grad(mask_net):
        noise = [
            param for name, param in mask_net.named_parameters() if "neuron_noise" in name
        ]
        for p in noise:
            p.grad.data = torch.sign(p.grad.data)

    @staticmethod
    def include_noise(mask_net):
        for m in mask_net.modules():
            if isinstance(m, NoisyBatchNorm2d):
                m.include_noise()

    @staticmethod
    def exclude_noise(mask_net):
        for m in mask_net.modules():
            if isinstance(m, NoisyBatchNorm2d):
                m.exclude_noise()

    @staticmethod
    def clip_mask(mask_net, lower=0.0, upper=1.0):
        params = [
            param for name, param in mask_net.named_parameters() if "neuron_mask" in name
        ]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)

    def mask_optim(
        self,
        mask_net,
        mask_optim,
        noise_optim,
        ds,
        anp_eps,
        **kwargs,
    ) -> None:
        for epoch in range(0, self.args.num_epochs + 1):
            mask_net.train()
            lr = mask_optim.param_groups[0]['lr']
            for _, (x, y) in enumerate(ds):
                x, y = x.to(self.args.device), y.to(self.args.device)
                # cal perturbation
                if anp_eps > 0.0:
                    self.reset(mask_net, True, anp_eps)
                    for _ in range(self.args.anp_steps):
                        noise_optim.zero_grad()
                        self.include_noise(mask_net)
                        output_noise = mask_net(x)
                        loss_noise = -F.cross_entropy(output_noise, y)
                        loss_noise.backward()
                        self.sign_grad(mask_net)
                        noise_optim.step()
                
                # optim mask
                # mask_optim.zero_grad()
            
                # if anp_eps > 0.0:
                #     self.include_noise(mask_net)
                #     output_noise = mask_net(x)
                #     loss_rob = F.cross_entropy(output_noise, y)
                # else:
                #     loss_rob = 0.0
                # self.exclude_noise(mask_net)
                # output_clean = mask_net(x)
                # loss_nat = F.cross_entropy(output_clean, y)
                # loss = self.args.anp_alpha * loss_nat + (1 - self.args.anp_alpha) * loss_rob
                # loss.backward()
                # mask_optim.step()
                # self.clip_mask(mask_net)

      

    def init_defense_utils(self) -> dict:
        model_mask_param = [
            p for n, p in list(self.mask_net.named_parameters()) if "neuron_mask" in n
        ]
        mask_optim = SGD(
            model_mask_param,
            lr=self.args.mask_optim_lr,
            momentum=0.9,
            weight_decay=0,
        )
        model_noise_param = [
            p for n, p in list(self.mask_net.named_parameters()) if "neuron_noise" in n
        ]
        noise_optim = SGD(
            model_noise_param,
            lr=self.args.noise_optim_lr,
            momentum=0,
            weight_decay=0,
        )
        criterion = torch.nn.CrossEntropyLoss().to(device)
        return (mask_optim, noise_optim, criterion)
   
    
    def do_expose(self):
        self.logger.info("="*20 + "Expose strategy: %s" % (self.args.discription) + "="*20)
        mask_optim, noise_optim, criterion = self.init_defense_utils()
        _ = deepcopy(self.mask_net.state_dict())

        for anp_eps in self.args.anp_eps:
            if anp_eps == 0:
                full_acc = self.accasr_full_test(self.mask_net, self.defense_loader, self.bad_test_loader, anp_eps, device=device)
                print('full_acc:', full_acc)
            else:

                self.mask_optim(
                    self.mask_net, mask_optim, noise_optim, self.defense_loader, cl_test=self.clean_test_loader, po_test=self.bad_test_loader, anp_eps=anp_eps
                )
                full_acc = self.accasr_full_test(self.mask_net, self.defense_loader, self.bad_test_loader, anp_eps, device=device)
                print('full_acc:', full_acc)

    
    def acctest(self, net, data_loader, device):
        net.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
                total_samples += len(labels)
        acc = total_correct / total_samples
        return acc if isinstance(acc, float) else acc.item() 

    def accasr_full_test(self, net, defense_loader, bad_test_loader, anp_eps, device=device) -> dict:
        num_cls = self.args.num_classes
        net.eval()
        ret = dict()
        correct_counter = [0] * num_cls
        cls_count = [0] * num_cls
        pred_counter = [0] * num_cls
        for _, (x, y) in enumerate(defense_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = net(x)
                # pred = output.argmax(dim=1)
                _, pred = output.max(1)
                for pred_ in pred:
                    pred_counter[pred_] += 1 
                for _idx, _y in enumerate(y):
                    cls_count[_y] += 1
                    # pred_counter[pred[_idx]] += 1
                    if pred[_idx] == _y:
                        correct_counter[_y] += 1
        ret["anp_eps"] = anp_eps
 
        ret["acc"] = round(sum(correct_counter) / sum(cls_count), 2)
        ret["asr"] = round(self.acctest(net, bad_test_loader, device), 2)
        # ret["cls_acc"] = [
        #     0 if _t == 0 else round(_c / _t,2) for _c, _t in zip(correct_counter, cls_count)
        # ]
        ret["cls_pred"] = [round(_ / sum(cls_count),2) for _ in pred_counter]
        # ret["cls_pred"] = [_ for _ in pred_counter]
        
        return ret