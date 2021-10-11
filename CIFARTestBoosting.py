from __future__ import print_function
import torch
from torchvision import datasets, transforms
import numpy as np
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18, WideResNet
import matplotlib.pyplot as plt
import os
from datetime import datetime
from Testing import testEnsemble
from Boosting import Ensemble, runBoosting
from AdversarialAttacks import attack_fgsm, attack_pgd
import json

import utils

import argparse

cuda = torch.device('cuda:0')

epsilons = [0.127]

# Training the ensemble

def test_ensemble(test_config):
#     resultsPath = f"results/plots/{test_config['training_method']}/{test_config['dataset_name']}/boosting/{test_config['num_samples_wl']}Eps{test_config['train_eps_wl']}/"
#     test_config['results_path'] = resultsPath
    resultsPath = test_config["results_path"]
    if not os.path.exists(resultsPath):
        os.mkdir(resultsPath)
    ensemble = testEnsemble(test_config)
    
    acc_file = resultsPath + f"acc_maxSamples_{test_config['num_samples_wl']}.png"
    adv_acc_file = resultsPath + f"adv_acc_maxSamples_{test_config['num_samples_wl']}.png"
    loss_file = resultsPath + f"loss_maxSamples_{test_config['num_samples_wl']}.png"
    ensemble.plot_accuracies(acc_file)
    ensemble.plot_loss(loss_file)
    ensemble.plot_adversarial_accuracies(adv_acc_file)
    return ensemble


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--num_samples_wl", default=30000, type=int)
    # parser.add_argument("--num_samples_train", default=200, type=int)
    # parser.add_argument("--num_samples_val", default=1500, type=int)
    # parser.add_argument("--train_eps_wl", default=8, type=int)
    # parser.add_argument("--num_wl", default=15, type=int) # CHANGE
    # parser.add_argument("--batch_size", default=128, type=int) # can try increasing this a lot
    # parser.add_argument("--attack_iters", default=20, type=int)
    # parser.add_argument("--testing_restarts", default=10, type=int)
    # parser.add_argument("--dataset_name", default='cifar10')
    parser.add_argument("--config_file", default="Configs/wongCIFAR10Test.json")
    parser.add_argument("--attack_name", default = "")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    test_config = {}

    with open(args.config_file) as f:
        test_config = json.load(f)
    

    if test_config['dataset_name'] == 'cifar10':
        test_config['dataset'] = datasets.CIFAR10
    elif test_config['dataset_name'] == 'cifar100':
        test_config['dataset'] = datasets.CIFAR100
        
    if len(args.attack_name):
        test_config['attack_name'] = args.attack_name

    # TODO? change these to args?
    test_config['weak_learner_type'] = WongBasedTrainingCIFAR10
    test_config['val_attacks'] = [attack_pgd]
    test_config['model_base'] = PreActResNet18
    test_config['path'] = f"./models/{test_config['training_method']}/{test_config['dataset_name']}/{test_config['num_samples_wl']}Eps{test_config['train_eps_wl']}/"
    # test_config['attack_eps_ensemble'] = [0.127]

    ensemble = test_ensemble(test_config)