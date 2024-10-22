import mlconfig
import torch
from . import ResNet, ToyModel, ISBBA_resnet, dynamic_models, ResNetWithAT
from torch.nn import CrossEntropyLoss

# Register mlconfig
# mlconfig.register(DatasetGenerator)
# Models
mlconfig.register(ResNet.ResNet)
mlconfig.register(ResNet.ResNet18)
mlconfig.register(ResNet.ResNet34)
mlconfig.register(ResNet.ResNet50)
mlconfig.register(ResNet.ResNet101)
mlconfig.register(ResNet.ResNet152)
mlconfig.register(ToyModel.ToyModel)
mlconfig.register(ISBBA_resnet.resnet18_200)
mlconfig.register(ResNetWithAT.ResnetWithAT)
# optimizer
mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)
