from typing import Callable, OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.optim import SGD, Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from .awp import AWPDefense
from .base import BasicDefense, EpochBasedDefense
from .fine_prune import FinePruneDefense
from .fine_tune import FineTuneDefense, UnlearnDefense

# =============================================================================
# Backdoor Neuron Expose
# =============================================================================

AWPDefense = AWPDefense
UnlearnDefense = UnlearnDefense
FinePruneDefense = FinePruneDefense


def weight_merge(pnet: nn.Module, alpha: float) -> None:
    _sd_ori = pnet.state_dict()
    for m in pnet.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)
    _sd_init = pnet.state_dict()
    _sd_new = dict()
    for (_k_ori, _v_ori), (_k_init, _v_init) in zip(_sd_ori.items(), _sd_init.items()):
        # TODO: how to merge
        assert _k_ori == _k_init
        _sd_new[_k_ori] = alpha * _v_ori + (1 - alpha) * _v_init
    _sd_new = OrderedDict(_sd_new)
    pnet.load_state_dict(_sd_new)


class OODFinetuneDefense(FineTuneDefense):
    def __init__(self, device: torch.device, num_epoch: int, **kwargs) -> None:
        super().__init__(device, num_epoch, **kwargs)
        self.noise_gen: Callable = kwargs.get("noise_gen", self.guassian_noise_gen)

    @staticmethod
    def guassian_noise_gen(x: torch.Tensor, **kwargs):
        return torch.randn_like(x)

    def do_defense(self, pnet: nn.Module, **kwargs) -> None:
        pnet.train()
        optim: Optimizer = kwargs.get("optim", None)
        sched: lr_scheduler._LRScheduler = kwargs.get("sched", None)
        ds: DataLoader = kwargs.get("ds", None)
        for _, (x, y) in enumerate(ds):
            # add noise to label
            x = x + self.noise_gen(x=x, y=y, **kwargs)
            x, y = x.to(self.device), y.to(self.device)
            optim.zero_grad()
            loss = self.alpha * self.criterion(pnet(x), y)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(  # type: ignore
                    pnet.parameters(),
                    max_norm=self.grad_clip_max,
                    norm_type=self.grad_clip_norm_type,
                )
            loss.backward()  # type: ignore
            if self.after_loss_backward_hook:
                self.after_loss_backward_hook(pnet=pnet, **kwargs)
            optim.step()
            if self.after_optim_step_hook:
                self.after_optim_step_hook(pnet=pnet, **kwargs)
        sched.step()


class RandomLabelFinetuneDefense(FineTuneDefense):
    """
    Finetuning the backdoored model in a small portion of label-shuffled clean data.
    """

    def do_defense(self, pnet: nn.Module, **kwargs) -> None:
        pnet.train()
        optim: Optimizer = kwargs.get("optim", None)
        sched: lr_scheduler._LRScheduler = kwargs.get("sched", None)
        ds: DataLoader = kwargs.get("ds", None)
        for _, (x, y) in enumerate(ds):
            # generate random one-hot
            y = torch.zeros_like(y).scatter_(1, torch.randint(0, 10, (y.size(0), 1)), 1)
            x, y = x.to(self.device), y.to(self.device)
            optim.zero_grad()
            loss = self.alpha * self.criterion(pnet(x), y)
            if self.grad_clip:
                nn.utils.clip_grad_norm_(  # type: ignore
                    pnet.parameters(),
                    max_norm=self.grad_clip_max,
                    norm_type=self.grad_clip_norm_type,
                )
            loss.backward()  # type: ignore
            if self.after_loss_backward_hook:
                self.after_loss_backward_hook(pnet=pnet, **kwargs)
            optim.step()
            if self.after_optim_step_hook:
                self.after_optim_step_hook(pnet=pnet, **kwargs)
        sched.step()


class WeightMagnitudePrune(BasicDefense):
    """
    Prunin the clean neurons w.r.t weight magnitude.
    """

    def __init__(self, device: torch.device, **kwargs) -> None:
        super().__init__(device, **kwargs)
        self.module_type: nn.Module = kwargs.get("module_type", nn.BatchNorm2d)
        self.thres = kwargs.get("thres", 0.0)

    def init_defense_utils(self, pnet: nn.Module, **kwargs) -> dict:
        return dict()

    def do_defense(self, pnet: nn.Module, **kwargs) -> None:
        pnet.eval()
        with torch.no_grad():
            for m in pnet.modules():
                if isinstance(m, self.module_type):  # type: ignore
                    m.weight.data[abs(m.weight.data) < self.thres] = 0.0  # type: ignore


class RKDDefense(EpochBasedDefense):
    """
    Maximize KL-devergence between two backdoored student model and a clean teacher model.
    """

    def __init__(self, device: torch.device, num_epoch: int, **kwargs) -> None:
        super().__init__(
            device,
            num_epoch,
            collecter_tag="rdk",
            ckpt_tag="rdk",
            ckpt_save_every=1,
            info_tag="rdk",
            **kwargs,
        )
        self.lr: float = kwargs.get("lr", 0.01)
        self.momentum: float = kwargs.get("momentum", 0.9)
        self.wd: float = kwargs.get("wd", 5e-4)

    def init_defense_utils(self, pnet: nn.Module, **kwargs) -> dict:
        optim = SGD(
            pnet.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd
        )
        return dict(optim=optim)

    def do_defense(self, pnet: nn.Module, tnet: nn.Module, **kwargs) -> None:
        optim: Optimizer = kwargs.get("optim", None)
        ds: DataLoader = kwargs.get("ds", None)
        tnet.eval()
        for p in tnet.parameters():
            p.requires_grad = False
        pnet.train()
        for _, (x, y) in enumerate(ds):
            x, y = x.to(self.device), y.to(self.device)
            optim.zero_grad()
            with torch.no_grad():
                tnet_y = tnet(x)
            with torch.enable_grad():
                pnet_y = pnet(x)
            loss = -F.kl_div(
                torch.softmax(pnet_y, dim=-1).log(),
                tnet_y.softmax(dim=-1),
                reduction="mean",
            )
            loss.backward()
            optim.step()


class EDRDefense(BasicDefense):
    def __init__(self, device: torch.device, **kwargs) -> None:
        super().__init__(device, **kwargs)

    def init_defense_utils(self, pnet: nn.Module, **kwargs) -> dict:
        return dict()

    def do_defense(self, pnet: nn.Module, **kwargs) -> None:
        pass