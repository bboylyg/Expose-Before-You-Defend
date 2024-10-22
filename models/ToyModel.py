import torch.nn as nn
import torch.nn.functional as F


class ConvBrunch(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(ConvBrunch, self).__init__()
        padding = (kernel_size - 1) // 2
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                      padding=padding),
            nn.BatchNorm2d(out_planes),
            nn.ReLU())

    def forward(self, x):
        features = self.out_conv(x)
        return features


class ToyModel(nn.Module):
    def __init__(self, block_size=1, layer_size=1, width=32,
                 feature_dim=512, num_classes=10):
        super(ToyModel, self).__init__()
        self.stem_conv = nn.Conv2d(3, width, kernel_size=3, padding=1)
        blocks = []
        for i in range(block_size):
            for j in range(layer_size):
                blocks.append(ConvBrunch(width, width, 3))
            blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.blocks = nn.Sequential(*blocks)
        self.f = nn.Linear(3*32*32, num_classes)
        self.fc = nn.Linear(3*32*32, num_classes)
        self.get_features = False

    def forward(self, x):
        # x = self.stem_conv(x)
        # x = self.blocks(x)
        # x = x.mean(dim=[2, 3])
        x = self.fc(x.view(x.shape[0], -1))
        # features = x
        # x = self.fc(x)
        # if self.training or self.get_features:
        #     return features, x
        return x
