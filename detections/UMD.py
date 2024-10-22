from __future__ import absolute_import
from __future__ import print_function

import argparse
import math
import os
import sys
import time
from copy import deepcopy
from random import random
from scipy import special
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import sys
import os
print(os.getcwd())
sys.path.append("..")
print(os.getcwd())
from Multi_Trigger_Backdoor_Attacks.model_for_cifar.model_select import select_model


# from Multi_Trigger_Backdoor_Attacks.datasets import mixed_backdoor_cifar
# from Multi_Trigger_Backdoor_Attacks.datasets.mixed_backdoor_cifar import add_Mtrigger_cifar, split_dataset
# sys.path.append("Backdoor_detection")


def data():
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
    ])

    clean_train = CIFAR10(root="./data", train=True, download=True, transform=transform_train)  # ,collate_fn=coffate
    clean_test = CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    clean_train, clean_val = split_dataset(clean_train, frac=0.01)

    return clean_train, clean_val, clean_test


def split_dataset(dataset, frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_split = int(frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_split:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_split:]].tolist()

    # generate the test set
    split_set = deepcopy(dataset)
    split_set.data = split_set.data[perm[:nb_split]]
    split_set.targets = np.array(split_set.targets)[perm[:nb_split]].tolist()

    print('total data size: %d images, split test size: %d images, split ratio: %f' % (
        len(train_set.targets), len(split_set.targets), frac))

    return train_set, split_set


def pert_est_class_pair(source, target, model, images, labels, pi=0.9, lr=1e-4, NSTEP=1000, init=None, verbose=False,
                        device='cuda'):
    if verbose:
        print("Perturbation estimation for class pair (s, t)".format(source, target))

    # Initialize perturbation
    if init is not None:
        pert = init
    else:
        pert = torch.zeros_like(images[0]).to(device)
    pert.requires_grad = True

    for iter_idx in range(NSTEP):

        # Optimizer: SGD
        optimizer = torch.optim.SGD([pert], lr=lr, momentum=0)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Get the loss
        images_perturbed = torch.clamp(images + pert, min=0, max=1)
        outputs = model(images_perturbed)
        loss = criterion(outputs, labels)

        # Update perturbation
        model.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step()

        # Compute misclassification fraction rho
        misclassification = 0
        with torch.no_grad():
            images_perturbed = torch.clamp(images + pert, min=0, max=1)
            outputs = model(images_perturbed)
            predicted = outputs.data.max(1)[1]
            misclassification += predicted.eq(labels).sum().item()
            rho = misclassification / len(labels)

        if verbose:
            print("current misclassification: {}; perturbation norm: {}".format(rho, torch.norm(
                pert).detach().cpu().numpy()))

        # Stopping criteria
        if rho >= pi or torch.norm(pert) > 20:
            break

    return pert.detach().cpu(), rho


def compute_score_combined(P, A, A_single, mode='mean'):
    # Get the weights for core nodes
    core = P[0]
    core_size = len(core)
    if core_size < 2:
        sys.exit('Core node less than 2!')
    A_core = A[core, :]
    A_core = A_core[:, core]
    # print(core)
    if mode == 'mean':
        score_core = np.sum(A_core) / (core_size * (core_size - 1))
    elif mode == 'min':
        score_core = np.min(np.sum(A_core, axis=1) / (core_size - 1))
    else:
        sys.exit('Wrong mode!')

    # Get the max connection for periphery nodes
    A_preph = A_single[core, :]
    A_preph = np.delete(A_preph, core, axis=1)
    score_preph = np.max(np.mean(A_preph, axis=0))

    return score_core - score_preph, score_core, score_preph


def get_threshold(N, conf=0.95):
    return np.sqrt(2) * special.erfinv(2 * np.power(conf, (1 / N)) - 1)


def umd(model, args):
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    config = {}
    ckpt_path = args.ckpt_path
    start_time = time.time()
    TRIAL = args.TRIAL
    NI = args.NI
    config["NUM_CLASS"] = args.num_classes
    PI = args.PI

    LR = args.LR
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
    ])
    detectset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    NC = config["NUM_CLASS"]

    correct_path = os.path.join(ckpt_path, "correct.npy")
    target_path = os.path.join(ckpt_path, "targets.npy")

    if os.path.exists(correct_path) and os.path.exists(target_path):
        print("Loading correctly classified images")
        correct = np.load(correct_path)
        targets = np.load(target_path)
    else:
        imgs = []
        labels = []
        index = []
        for i in range(len(detectset.targets)):
            sample, label = detectset.__getitem__(i)
            imgs.append(sample)
            labels.append(label)
            index.append(i)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        index = torch.tensor(index)
        correct = []
        targets = []
        bs = args.batch_size
        for img, label, i in zip(imgs.chunk(math.ceil(len(imgs) / bs)),
                                 labels.chunk(math.ceil(len(imgs) / bs)), index.chunk(math.ceil(len(imgs) / bs))):
            img = img.to(device)
            target = label.to(device)
            i = i.to(device)
            with torch.no_grad():
                outputs = model(img)
                predicted = outputs.data.max(1)[1]
            correct.extend(i[predicted.eq(target)].cpu().numpy())
            targets.extend(target[predicted.eq(target)].cpu().numpy())

    np.save(os.path.join(ckpt_path, "correct.npy"), correct)
    np.save(os.path.join(ckpt_path, "targets.npy"), targets)
    images_all = []
    ind_all = []
    for c in range(NC):
        ind = [correct[i] for i, label in enumerate(targets) if label == c]
        ind = np.random.choice(ind, NI, replace=False)
        images_all.append(torch.stack([detectset[i][0] for i in ind]))
        ind_all.append(ind)
    images_all = [images.to(device) for images in images_all]
    np.save(os.path.join(ckpt_path, 'ind.npy'), ind_all)
    for s in range(NC):
        for t in range(NC):
            # skip the case where s = t
            if s == t:
                continue
            images = images_all[s]
            labels = (torch.ones((len(images),)) * t).long().to(device)

            # CORE STEP: perturbation esitmation for (s, t) pair
            norm_best = 1000000.
            pattern_best = None
            pert_best = None
            mask_best = None
            rho_best = None
            for trial_run in range(TRIAL):

                pert, rho = pert_est_class_pair(source=s, target=t, model=model, images=images, device=device,
                                                labels=labels, pi=PI, lr=LR, init=None, verbose=False)
                if torch.norm(pert) < norm_best:
                    norm_best = torch.norm(pert)
                    pert_best = pert
                    rho_best = rho

            print(s, t, torch.norm(pert).item(), rho)
            torch.save(pert.detach().cpu(), os.path.join(ckpt_path, 'pert_{}_{}'.format(s, t)))
    print("--- %s seconds ---" % (time.time() - start_time))
    torch.save((time.time() - start_time), os.path.join(ckpt_path, 'time'))

    correct_path = os.path.join(ckpt_path, "correct.npy")
    target_path = os.path.join(ckpt_path, "targets.npy")
    if os.path.exists(correct_path) and os.path.exists(target_path):
        print("Loading correctly classified images")
        correct = np.load(correct_path)
        targets = np.load(target_path)
    else:
        imgs = []
        labels = []
        index = []
        for i in range(len(detectset.targets)):
            sample, label = detectset.__getitem__(i)
            imgs.append(sample)
            labels.append(label)
            index.append(i)
        imgs = torch.stack(imgs)
        labels = torch.tensor(labels)
        index = torch.tensor(index)
        correct = []
        targets = []

        bs = args.batch_size
        for img, label, i in zip(imgs.chunk(math.ceil(len(imgs) / bs)),
                                 labels.chunk(math.ceil(len(imgs) / bs)), index.chunk(math.ceil(len(imgs) / bs))):
            img = img.to(device)
            target = label.to(device)
            i = i.to(device)
            with torch.no_grad():
                outputs = model(img)
                predicted = outputs.data.max(1)[1]
            correct.extend(i[predicted.eq(target)].cpu().numpy())
            targets.extend(target[predicted.eq(target)].cpu().numpy())
    images_all = []
    ind_all = []
    for c in range(NC):
        ind = [correct[i] for i, label in enumerate(targets) if label == c]
        ind = np.random.choice(ind, NI, replace=False)
        images_all.append(torch.stack([detectset[i][0] for i in ind]))
        ind_all.append(ind)

    for t in range(NC):
        for s in range(NC):
            if s == t:
                continue
            # Get the estimated perturbation

            pert = torch.load(os.path.join(ckpt_path, 'pert_{}_{}'.format(s, t))).to(device)

            acc_map = torch.zeros((NC, NC))

            for s_trans in range(NC):
                images = images_all[s_trans].to(device)
                with torch.no_grad():

                    images_perturbed = torch.clamp(images + pert, min=0, max=1)

                    outputs = model(images_perturbed)
                    predicted = outputs.data.max(1)[1]
                freq = torch.zeros((NC,))
                predicted = predicted.cpu()
                for i in range(len(freq)):
                    freq[i] = len(np.where(predicted == i)[0])
                freq[s_trans] = 0
                if s_trans == s:
                    freq[t] = 0
                acc_map[s_trans, :] = freq / NI
            acc_map = acc_map.detach().cpu().numpy()
            torch.save(acc_map, os.path.join(ckpt_path, 'color_map_{}_{}'.format(s, t)))

    N_init = args.N_init
    poisoned_pairs = args.poisoned_pairs
    print("Expected pairs: ")
    print(poisoned_pairs)
    scores = []
    pairs_set = []
    theta_set = []
    # Get adjacency matrix
    stat_evals = []
    H_score = []
    pert_null = []
    pert_eval = []
    pairs = []
    trans_graph = []
    for t in range(NC):
        for s in range(NC):
            pairs.append([s, t])
            if s != t:
                trans = torch.load(os.path.join(ckpt_path, 'color_map_{}_{}'.format(s, t)))
                trans = np.transpose(trans)  # The color map is originally stored with each role the same source class
            else:
                trans = np.zeros((NC, NC))
            trans = np.reshape(trans, (NC * NC))  # (0, 0), (0, 1), ..., (0, 9), (1, 0), (1, 1), ...
            trans_graph.append(trans)
    pairs = np.asarray(pairs)
    trans_graph = np.asarray(trans_graph)
    trans_graph_single = trans_graph
    trans_graph_mutual = (trans_graph + np.transpose(trans_graph)) / 2

    # Remove class pairs (0, 0), (1, 1), ...
    idx = 0
    idx_remove = []
    for i in range(len(pairs)):
        if pairs[i][0] == pairs[i][1]:
            idx_remove.append(idx)
        idx += 1
    # Reshape graph
    trans_graph_single = np.delete(trans_graph_single, idx_remove, axis=0)
    trans_graph_single = np.delete(trans_graph_single, idx_remove, axis=1)
    trans_graph_mutual = np.delete(trans_graph_mutual, idx_remove, axis=0)
    trans_graph_mutual = np.delete(trans_graph_mutual, idx_remove, axis=1)
    pairs = np.delete(pairs, idx_remove, axis=0)  # remove labels of class pairs (0, 0), (1, 1), ...

    # Vertices, edges
    V_names = pairs
    A_single = trans_graph_single
    A_mutual = trans_graph_mutual
    V = np.arange(start=0, stop=len(V_names), dtype=int)

    # Get initial community
    A_flatten = A_mutual.flatten()
    rank = np.flip(np.argsort(A_flatten))
    init_candidate = []
    for i in range(len(rank)):
        pair1 = int(rank[i] / len(V))
        pair2 = int(rank[i] % len(V))
        pair1_name = V_names[pair1]
        pair2_name = V_names[pair2]
        if pair1_name[0] != pair2_name[0]:  # Source class should be different
            init_candidate.append([pair1, pair2])
        if len(init_candidate) == N_init:
            break

    core_best_global = None
    score_best_global = - float("inf")
    for init in init_candidate:
        # Initialize the core-periphery structure
        core_record = [np.array(init)]
        P = [np.array(init), np.delete(V, np.array(init))]
        score, score_mul, score_sin = compute_score_combined(P, A_mutual, A_single, mode='min')
        score_record = [score]
        score_mul_record = [score_mul]
        score_sin_record = [score_sin]
        converge = False
        while not converge:
            core_best = None
            score_best = - float("inf")
            # Trial include every node in the periphery to the core
            core_old = core_record[-1]
            preph_old = np.delete(V, core_old)
            for i in range(len(preph_old)):
                # Skip if the source class already exists
                s = V_names[preph_old[i]][0]
                skip = False
                for j in range(len(core_old)):
                    if s == V_names[core_old[j]][0]:
                        skip = True
                        break
                if skip:
                    continue
                core_trial = np.concatenate([core_old, [preph_old[i]]])
                preph_trial = np.delete(preph_old, i)
                P_trial = [core_trial, preph_trial]
                score_trial, score_mul, score_sin = compute_score_combined(P_trial, A_mutual, A_single, mode='min')
                if score_trial > score_best:
                    score_best = score_trial
                    core_best = core_trial
                    score_mul_best = score_mul
                    score_sin_best = score_sin
            if core_best is None:
                converge = True
            else:
                core_record.append(core_best)
                score_record.append(score_best)
                score_mul_record.append(score_mul_best)
                score_sin_record.append(score_sin_best)
        if np.max(score_record) > score_best_global:
            score_best_global = np.max(score_record)
            core_best_global = core_record[np.argmax(score_record)]
            core_last = core_record[-1]

    core = core_best_global
    preph = np.delete(V, core)
    P = [core, preph]
    score = score_best_global
    H_score.append(score)
    pairs_detected = pairs[core]
    print("Detected pairs: ")
    print((pairs_detected.tolist()))
    np.save(os.path.join(ckpt_path, 'pairs_detected.npy'), pairs_detected)
    # Plot sorted adjacency matrix
    order = []
    for i in range(len(P)):
        for j in range(len(P[i])):
            order.append(P[i][j])
    A_single = A_single[order, :]
    A_single = A_single[:, order]
    plt.imshow(A_single, cmap='hot', vmin=0, vmax=1)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(ckpt_path, 'color_map_all.png'))
    plt.close()
    pairs_detected = np.load(os.path.join(ckpt_path, 'pairs_detected.npy'))
    pairs_idx = pairs_detected[:, 0] * NC + pairs_detected[:, 1]
    pairs_set.append(pairs_idx)

    score = []
    # Get pert/patch norm
    pert_size_eval = []
    pert_size_null = []
    for t in range(NC):
        for s in range(NC):
            if s != t:
                pert = torch.load(os.path.join(ckpt_path, 'pert_{}_{}'.format(s, t)))
                pert_size = torch.norm(pert).item()
                if np.where(pairs_idx == s * NC + t)[0] > 0:
                    pert_size_eval.append(pert_size)
                else:
                    pert_size_null.append(pert_size)
    stat_eval = np.asarray(pert_size_eval)
    stat_null = np.asarray(pert_size_null)

    pert_eval.append(pert_size_eval)
    pert_null.append(pert_size_null)

    stat_eval = 1 / stat_eval
    stat_eval = np.median(stat_eval)
    stat_evals.append(stat_eval)
    stat_null = 1 / stat_null

    med = np.median(stat_null)
    MAD = np.median(np.abs(stat_null - med))
    scores.append((stat_eval - med) / (MAD * 1.4826))
    theta = get_threshold(len(stat_null), conf=0.95)
    theta_set.append(theta)

    print("Null size statistic")
    print([np.median(pert_size_null) for pert_size_null in pert_null])
    print("Eval size statistic")
    print([np.median(pert_size_eval) for pert_size_eval in pert_eval])
    print("Pert reverse size stat is")
    print(stat_evals)
    print("Threshold is: ")
    print(theta_set)
    print("Score is: ")
    print(scores)
    num_detected = 0
    ind = []
    for i in range(len(scores)):
        if scores[i] > theta_set[i]:
            num_detected += 1
    print("Number of detected models: {}".format(num_detected))

    stat_evals = np.asarray(stat_evals)
    H_score = np.asarray(H_score)
    theta_set = np.asarray(theta_set)
    scores = np.asarray(scores)
    pairs_set = np.asarray(pairs_set)

    stat = scores
    pairs_idx = pairs_set[np.argmax(stat)]
    poisoned_pairs = np.array(poisoned_pairs)
    poisoned_pairs_idx = poisoned_pairs[:, 0] * NC + poisoned_pairs[:, 1]
    num = 0
    for i in pairs_idx:
        if i in poisoned_pairs_idx:
            num += 1
    print("# of detected pairs: {}".format(num))


    if num_detected:  # 判断是否为后门模型
        print("This is a backdoor model")
    else:
        print("Not a backdoor model")

    return num_detected


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--NSTEP', default=3000)
    parser.add_argument('--model_path',
                        default='../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar')
    parser.add_argument('--ckpt_path', default='umd')
    parser.add_argument('--TRIAL', default=1)
    parser.add_argument('--NI', default=10)
    parser.add_argument('--PI', default=0.9)
    parser.add_argument('--LR', default=1e-5)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--N_init', default=5)
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--model_name', default='ResNet18',
                        choices=['ResNet18', 'VGG16', 'PreActResNet18', 'MobileNetV2'])
    parser.add_argument('--pretrained', default=True,help='read model weight')

    parser.add_argument('--multi_model', default=False,help="Detect the model in the Model folder")
    parser.add_argument('--model_name_list', '--arg',default=[], nargs='+', help="model architecture")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.multi_model == False:
        model_path = os.path.join(args.model_path)
        pretrained_models_path = model_path
        model = select_model(args=args, dataset=args.dataset, model_name=args.model_name, pretrained=args.pretrained,
                             pretrained_models_path=pretrained_models_path)
        poisoned_pairs = []
        for num_classe_1 in range(args.num_classes):
            for num_classe_2 in range(args.num_classes):
                if num_classe_2 != num_classe_1:
                    poisoned_pairs.append([num_classe_2, num_classe_1])
        args.poisoned_pairs = poisoned_pairs

        num_detected = umd(args=args, model=model)

    else:
        model_name_list = args.model_name_list
        out = []
        path = os.path.split(os.path.realpath(__file__))[0]  # 当前路径
        models = os.listdir(path + '/../model')  # 模型路径
        for i in range(len(models)):
            for j in model_name_list:
                if models[i].startswith(j) and args.dataset in models[i]:
                    poisoned_pairs = []
                    for num_classe_1 in range(args.num_classes):
                        for num_classe_2 in range(args.num_classes):
                            if num_classe_2 != num_classe_1:
                                poisoned_pairs.append([num_classe_2, num_classe_1])
                    args.poisoned_pairs = poisoned_pairs

                    args.model_path = '../model/' + models[i]
                    model_path = os.path.join(args.model_path)
                    pretrained_models_path = model_path
                    print("~~~~~~~~~~~~~~ UMD ~~~~~~~~~~~~~~")
                    print(models[i])
                    print()
                    model = select_model(args=args, dataset=args.dataset, model_name=j, pretrained=args.pretrained,
                                         pretrained_models_path=pretrained_models_path)
                    out.append(umd(args=args, model=model))
        print("~~~~~~~~~~~~~~ UMD ~~~~~~~~~~~~~~")
        print(out)


