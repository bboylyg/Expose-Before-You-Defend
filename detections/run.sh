#NC detection
#simple model
python NC.py --dataset CIFAR10 --model_name ResNet18 --model_path ../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar
#multi-model
python NC.py  --dataset CIFAR10 --model_name ResNet18 --multi_model True --model_name_list ResNet18 VGG16 PreActResNet18 MobileNetV2

#Unlearning detection
#simple model
python Unlearning.py --dataset CIFAR10 --model_name ResNet18 --model_path ../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar
#multi-model
python Unlearning.py  --dataset CIFAR10 --model_name ResNet18 --multi_model True --model_name_list ResNet18 VGG16 PreActResNet18 MobileNetV2

#MMBD detection
#simple model
python MMBD.py --dataset CIFAR10 --model_name ResNet18 --model_path ../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar
#multi-model
python MMBD.py  --dataset CIFAR10 --model_name ResNet18 --multi_model True --model_name_list ResNet18 VGG16 PreActResNet18 MobileNetV2

#UMD detection
#simple model
python UMD.py --dataset CIFAR10 --model_name ResNet18 --model_path ../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar
#multi-model
python UMD.py  --dataset CIFAR10 --model_name ResNet18 --multi_model True --model_name_list ResNet18 VGG16 PreActResNet18 MobileNetV2




