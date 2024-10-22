import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

class ResnetWithAT(ResNet):
    """
    https://github.com/SforAiDl/KD_Lib/blob/master/KD_Lib/models/resnet.py
    """
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        activation1 = x
        x = self.layer2(x)  # 16x16
        activation2 = x
        x = self.layer3(x)  # 8x8
        activation3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, activation1, activation2, activation3
