from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from WongBasedTraining import WongBasedTrainingCIFAR10
from PGDBasedTraining import PGDBasedTraining
from TradesBasedTraining import TradesBasedTrainingCIFAR10
from Architectures import PreActResNet18, PreActResNet18_100, WideResNet, WideResNet34_10_10, WideResNet34_100_10
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json
from utils import applyDSTrans

from AdversarialAttacks import attack_fgsm, attack_pgd

import utils

import argparse

cuda = torch.device('cuda:0')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="Configs/wongCIFAR10Train.json")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    train_config = {}

    with open(args.config_file) as f:
        train_config = json.load(f)

    if train_config['dataset_name'] == 'cifar10':
        train_config['dataset'] = datasets.CIFAR10
        train_config['model_base'] = PreActResNet18
    elif train_config['dataset_name'] == 'cifar100':
        train_config['dataset'] = datasets.CIFAR100
        train_config['model_base'] = PreActResNet18_100

    if train_config["training_method"] == "trades" and train_config["dataset_name"]=="cifar10":
        train_config["model_base"] = WideResNet34_10_10
    elif train_config["training_method"] == "trades" and train_config["dataset_name"] == "cifar100":
        train_config["model_base"] = WideResNet34_100_10

    if train_config['training_method'] == 'wong':
        train_config['weak_learner_type'] = WongBasedTrainingCIFAR10
    elif train_config['training_method'] == 'pgd':
        train_config['weak_learner_type'] = PGDBasedTraining
    elif train_config['training_method'] == 'trades':
        train_config['weak_learner_type'] = TradesBasedTrainingCIFAR10
    train_config['val_attacks'] = [attack_pgd]

    wl = train_config['weak_learner_type'](train_config['model_base'], train_config['attack_eps_wl'])
    train_ds, test_ds = applyDSTrans(train_config)
    train_ds.targets = torch.tensor(np.array(train_ds.targets))
    test_ds.targets = torch.tensor(np.array(test_ds.targets))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_config['batch_size_wl'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=train_config['batch_size_wl'], shuffle=True)
    
    wl.fit(train_loader, test_loader, train_config)
    
    