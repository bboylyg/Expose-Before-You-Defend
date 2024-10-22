# Examples of using OpenTAIBackdoor
The goal of our **OpenTAIBackdoor** is to provide a simple framework for researching adversarial attacks and defence.

---
## Quickstart examples
For a quick start, we have provided the following examples:
- train.py
- *others coming soon*

---
### Backdoor Attacks Training with OpenTAIBackdoor
Here are some descriptions of the provided [**train.py**](train.py) \
The experiments configuration files are stored under [configs folders](configs/). We provided the example using ResNet-18 model on CIFAR-10.
To run the example to train a backdoored model of `BadNets`, please follow the following:
```python
    python train.py --exp_path /PATH/TO/YOUR/EXPERIMENT/FOLDER \
                    --exp_name cifar10_badnet_rn18              \
                    --exp_configs configs/                     \
```
 - **--exp_name** option can be replaced with other experiments configuration files stored under the configs folder.
 - **--exp_path** is where you want to store the experiment's files, such as checkpoints and logs
 - **--exp_config** is the path to the folder where the configuration files are stored
 - **--data_parallel** if you want to use data_parallel **torch.nn.DataParallel**.
<!--  - Adv Attack Options (**--eps**, **--num_steps**, **--step_size**, **--attack_choice**) specifies the adversarial attack setting, this script will run a adversarial attack evaluation on validation set in every epochs to validate the performance. -->
