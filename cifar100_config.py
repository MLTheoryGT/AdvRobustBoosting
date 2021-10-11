import torch
from torchvision import datasets, transforms
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18_100
from Ensemble import Ensemble
from AdversarialAttacks import attack_pgd
import numpy as np
import os
import csv
from Testing import testEnsemble

train_config = {
    "num_samples_wl": 750031,
    "num_wl": 4,
    "train_eps_wl": 8,
    "training_valrestarts": 1,
    "dataset_name": "cifar100",
    "train_alpha_wl": 10,
    "lr_min_wl": 0,
    "lr_max_wl": 0.2,
    "weight_decay_wl": 1e-4,
    "early_stop_wl": True,
    "delta_init_wl": "random",
    "momentum_wl": 0.9,
    "seed_wl": None,
    "opt_level_wl": "O2",
    "loss_scale_wl": 1.0,
    "delta_init_wl": "random",
    "momentum_wl": 0.9,
    "attack_iters_val_wl": 20,
    "batch_size_wl": 128,
    "restarts_wl": 1,
    "training_method": "wong",
    "adv_train_wl": True,
    "prediction_weights_wl": False,
    "lr_schedule_wl": "cyclic",
    "attack_eps_wl": [0.127],
    "model_base": PreActResNet18_100,
    "val_attacks": [attack_pgd],
    "dataset": datasets.CIFAR100,
    "weak_learner_type": WongBasedTrainingCIFAR10
}

path_head = f"./models/{train_config['training_method']}/{train_config['dataset_name']}/baseline/{train_config['num_samples_wl']}Eps{train_config['train_eps_wl']}/"

test_config = {
    "num_samples_wl": 750031,
    "num_samples_train": 200,
    "num_samples_val": 10000,
    "train_eps_wl": 8,
    "num_wl": 15,
    "batch_size": 512,
    "testing_attack_iters": 20,
    "testing_restarts": 10,
    "dataset_name": "cifar100",
    "training_method": "wong",
    "train_batch_size": 128,
    "test_batch_size": 512,
    "attack_eps_ensemble": [0.127],
    "auto_attack": True,
    "model_base": PreActResNet18_100,
    "dataset": datasets.CIFAR100,
    'weak_learner_type': WongBasedTrainingCIFAR10,
    'path': path_head
}