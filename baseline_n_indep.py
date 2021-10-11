import torch
from torchvision import datasets, transforms
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18
from Ensemble import Ensemble
from AdversarialAttacks import attack_pgd
import numpy as np
import os
import csv
from Testing import testEnsemble

from cifar100_config import train_config, test_config


weakLearners = []
weakLearnerWeights = []

# from ipywidgets import IntProgress
from utils import applyDSTrans
train_ds, test_ds = applyDSTrans(train_config)
train_ds.targets = torch.tensor(np.array(train_ds.targets))
test_ds.targets = torch.tensor(np.array(test_ds.targets))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
path_head = f"./models/{train_config['training_method']}/{train_config['dataset_name']}/baseline/{train_config['num_samples_wl']}Eps{train_config['train_eps_wl']}/"

# if os.path.exists(path_head):
#     print("Already exists, exiting")
# else:
#     os.mkdir(path_head)
    

# for t in range(train_config['num_wl']):
#     h_i = train_config['weak_learner_type'](model_base=train_config['model_base'], attack_eps=train_config['attack_eps_wl'])
#     h_i.fit(train_loader, test_loader, train_config)
#     model_path = f'{path_head}wl_{t}.pth'
#     torch.save(h_i.model.state_dict(), model_path)
#     weakLearners.append(model_path)
#     print("weakLearnerWeights before:", weakLearnerWeights)
#     weakLearnerWeights.append(1.)
#     print("weakLearnerWeights after:", weakLearnerWeights)

# weight_path = f"{path_head}wl_weights.csv"
# with open(weight_path, 'w') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(weakLearnerWeights)

# test_config = {
#     "num_samples_wl": 750030,
#     "num_samples_train": 200,
#     "num_samples_val": 10000,
#     "train_eps_wl": 8,
#     "num_wl": 15,
#     "batch_size": 128,
#     "testing_attack_iters": 20,
#     "testing_restarts": 10,
#     "dataset_name": "cifar10",
#     "training_method": "wong",
#     "train_batch_size": 128,
#     "test_batch_size": 512,
#     "attack_eps_ensemble": [0.127],
#     "auto_attack": True,
#     "model_base": PreActResNet18,
#     "dataset": datasets.CIFAR10,
#     'weak_learner_type': WongBasedTrainingCIFAR10,
#     'path': path_head
# }

# test_config['results_path'] = f"results/plots/{test_config['training_method']}/{test_config['dataset_name']}/snapshot/{test_config['num_samples_wl']}Eps{test_config['train_eps_wl']}"
test_config['results_path'] = f"results/plots/{test_config['training_method']}/{test_config['dataset_name']}/baseline/{test_config['num_samples_wl']}Eps{test_config['train_eps_wl']}_full"

testEnsemble(test_config)