import torch
import numpy as np
from taibackdoor.models import dynamic_models
from torchvision import datasets
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def create_bd(netG, netM, inputs, targets, opt):
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output


class DynamicCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # Load models
        ckpt_path = 'trigger/dynamic_all2one_cifar10_ckpt.pth.tar'
        state_dict = torch.load(ckpt_path)
        opt = state_dict["opt"]
        netG = dynamic_models.Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        netM = dynamic_models.Generator(opt, out_channels=1).to(device)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.eval()

        # Add Backdoor Trigers
        if train is False:
            idx = np.array(self.targets) != target_label
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            poison_rate = 1.0

        s = len(self)
        self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.247, 0.243, 0.261])

        for i in self.poison_idx:
            x = self.data[i]
            y = self.targets[i]
            x = torch.tensor(x).permute(2, 0, 1) / 255.0
            x_in = torch.stack([normalizer(x)]).to(device)
            p, m = create_bd(netG, netM, x_in, y, opt)
            p = p[0, :, :, :].detach().cpu()
            m = m[0, :, :, :].detach().cpu()
            x_bd = x + (p - x) * m
            x_bd = x_bd.permute(1, 2, 0).numpy() * 255
            x_bd = x_bd.astype(np.uint8)
            self.data[i] = x_bd

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Injecting: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.2f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx),
               poison_rate))
