import numpy as np
import torch
import torch.nn as nn
import logging
import argparse
import os
import pandas as pd
from torch.optim import SGD, Adam, Optimizer, lr_scheduler


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Unlearning():
    def __init__(self, args, logger, net, data_loader, **kwargs) -> None:
        self.args = args
        self.arguments()
        self.logger = logger

        self.logger.info(self.args)

        self.net = net
        self.defense_loader = data_loader['defense_loader']
        self.clean_test_loader = data_loader['clean_test_loader']
        self.bad_test_loader = data_loader['bad_test_loader']

    def arguments(self):
        if 'discription' not in self.args:
            self.args.discription = 'Unlearning'
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'print_every' not in self.args:
            self.args.print_every = 500
        if 'num_classes' not in self.args:
            self.args.num_classes = 10
        if 'unlearn_epochs' not in self.args:
            self.args.unlearn_epochs = 20
        if 'lr' not in self.args:
            self.args.lr = 0.00001
        if 'sched_gamma' not in self.args:
            self.args.sched_gamma = 0.1
        if 'sched_ms' not in self.args:
            self.args.sched_ms = [20, 20]
        if 'stop_acc' not in self.args:
            self.args.stop_acc = 0.1
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def init_defense_utils(self) -> dict:
        # optim = SGD(
        #     self.net.parameters(),
        #     lr=self.args.lr,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )

        optim = Adam(self.net.parameters(),
                     lr=self.args.lr,
                     weight_decay=5e-4,
                     )

        sched = lr_scheduler.MultiStepLR(optim, self.args.sched_ms, gamma=self.args.sched_gamma)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        return (optim, sched, criterion)
    
    def do_expose(self):
        self.logger.info("="*20 + "Expose strategy: %s" % (self.args.discription) + "="*20)
        # save training progress
        test_process = []
        save_name_pre = self.args.output_logs_path + f'{self.args.dataset}_{self.args.arch}_{self.args.trigger_type}_{self.args.discription}.csv'
        
        if os.path.exists(save_name_pre):
            os.remove(save_name_pre)

        optim, sched, criterion = self.init_defense_utils()
        self.net.train()       
        for epoch in range(0, self.args.unlearn_epochs + 1):
            if epoch == 0:
            # before training test firstly
                
                lr = optim.param_groups[0]['lr']
                full_acc = self.accasr_full_test(self.net, self.defense_loader, self.bad_test_loader, epoch, lr, device=device)
                print('full_acc:', full_acc)

                test_process.append(
                    (epoch, lr, full_acc['acc'], full_acc['asr'], full_acc["cls_pred"]))

            else:   
                for i, (images, labels) in enumerate(self.defense_loader):
                    images, labels = images.to(device), labels.to(device)
                    optim.zero_grad()
                    output = self.net(images)
                    loss = criterion(output, labels)

                    nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=20, norm_type=2)
                    (-loss).backward()
                    optim.step()
            
                sched.step()
                lr = optim.param_groups[0]['lr']
                full_acc = self.accasr_full_test(self.net, self.defense_loader, self.bad_test_loader, epoch, lr, device=device)
    
                print('full_acc:', full_acc)

                test_process.append(
                    (epoch, lr, full_acc['acc'], full_acc['asr'], full_acc["cls_pred"]))
            
        df = pd.DataFrame(test_process, columns=("Epoch", "LR", "CA", "ASR",                                          "cls_pre"))
        df.to_csv(save_name_pre, mode='a', index=False, encoding='utf-8')


            # if full_acc["acc"] <= self.clean_threshold:
            #     # save the last checkpoint
            #     file_path = os.path.join(args.output_weight, f'unlearned_model_last.tar')
            #     # torch.save(net.state_dict(), os.path.join(args.output_dir, 'unlearned_model_last.tar'))
            #     save_checkpoint({
            #         'epoch': epoch,
            #         'state_dict': net.state_dict(),
            #         'clean_acc': cl_test_acc,
            #         'bad_acc': po_test_acc,
            #         'optimizer': optimizer.state_dict(),
            #     }, file_path)
            #     break
    
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

    def accasr_full_test(self, net, defense_loader, bad_test_loader, epoch, lr, device=device) -> dict:
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
        ret["epoch"] = epoch
        ret["lr"] = lr
        ret["acc"] = round(sum(correct_counter) / sum(cls_count), 2)
        ret["asr"] = round(self.acctest(net, bad_test_loader, device), 2)
        # ret["cls_acc"] = [
        #     0 if _t == 0 else round(_c / _t,2) for _c, _t in zip(correct_counter, cls_count)
        # ]
        ret["cls_pred"] = [round(_ / sum(cls_count),2) for _ in pred_counter]
        # ret["cls_pred"] = [_ for _ in pred_counter]
        
        return ret

