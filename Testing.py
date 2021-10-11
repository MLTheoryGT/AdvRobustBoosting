from Boosting import Ensemble
from WongBasedTraining import WongBasedTrainingCIFAR10
from Architectures import PreActResNet18, PreActResNet18_100
from AdversarialAttacks import attack_pgd
from torchvision import datasets, transforms
from datetime import datetime
import torch
from utils import applyDSTrans
import numpy as np
from autoattack import AutoAttack


"""
    path: should contain wl number of .pth models named cifar10_wl_i.pth
        should also contain a weights.csv of form "alpha_1, alpha_2, ..., alpha_n"
    attacks: pass in functions
    numWL: int
"""
# def testEnsemble(path, attacks, numWL, dataset=datasets.CIFAR10, numsamples_train=1000, numsamples_val=1000, attack_eps_ensemble = [0.03], attack_iters=20, restarts=10, gradOptWeights=False):

def testEnsemble(config):
    if config['dataset'] == datasets.CIFAR10:
        model_base = PreActResNet18
    elif config['dataset'] == datasets.CIFAR100:
        model_base = PreActResNet18_100
    train_ds, test_ds = applyDSTrans(config)
    train_ds.targets = torch.tensor(np.array(train_ds.targets))
    test_ds.targets = torch.tensor(np.array(test_ds.targets))
    
    
    #mini loaders for ensemble
    # @Arvind, I think you may be able to change the batch size here
    test_loader_mini = torch.utils.data.DataLoader(test_ds, batch_size=config['test_batch_size'], shuffle=False) #Note: change this to True when using a subset
    train_loader_mini = torch.utils.data.DataLoader(train_ds,
        batch_size=config['train_batch_size'], shuffle=True) #change to True?
    
    wl = []
    wlWeights = []
    with open(config['path'] + "wl_weights.csv") as f:
        for line in f:
            wlWeights = line.split(",")
            break
    
    for i in range(len(wlWeights)):
        wlWeights[i] = float(wlWeights[i])
    for i in range(len(wlWeights)):
        wl.append(config['path'] + f'wl_{i}.pth')
    
    startTime = datetime.now()

    ensemble = Ensemble(weakLearners=[], weakLearnerWeights=[], weak_learner_type=WongBasedTrainingCIFAR10, attack_eps=config['attack_eps_ensemble'], model_base=model_base)
    if config["dataset_name"] == "cifar10":
        forwardPass = ensemble.predictUnnormalizedDataCIFAR10
    elif config["dataset_name"] == "cifar100":
        forwardPass = ensemble.predictUnnormalizedDataCIFAR100
    
    if config["training_method"] == "trades":
        forwardPass = ensemble.predict
    
    num_batches = 8
    xys = [(x, y) for (x, y) in test_loader_mini]
    l = [x for (x, y) in xys]
    x_test = torch.cat(l, 0)
#     x_test = torch.cat(l[:num_batches], 0) for doing less than the whole test set
    l = [y for (x, y) in xys]
    y_test = torch.cat(l, 0)
#     y_test = torch.cat(l[:num_batches], 0)
    min_wl = 1

    for i in range(config['num_wl']):
        print("Weak Learner ", i, ".  Time Elapsed (s): ", (datetime.now()-startTime).seconds)
        ensemble.addWeakLearner(wl[i], wlWeights[i])
        if i < min_wl:
            continue
#         ensemble.addWeakLearner(wl[i], 1.0)
#         ensemble.gradOptWeights(train_loader_mini)
#         ensemble.addWeakLearner(wl[i], 0.01)
#         print("before ens acc", ensemble.accuracies)
        if config["auto_attack"]:
            adversary = AutoAttack(forwardPass, norm='Linf', eps=0.031, version='standard', log_path=config['results_path'] + f"log_wl_{i}.txt")
            adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
            x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=config["test_batch_size"])
            
        
#             Below is logic for attacking all examples with each attack
#             attacks = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
#             for attack in attacks:
#                 # NOTE: hardcoded eps=0.031
#                 adversary = AutoAttack(forwardPass, norm='Linf', eps=0.031, version='standard', log_path=config['results_path'] + f"log_wl_{i}_indiv.txt")
#                 adversary.attacks_to_run = [attack]
#                 x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=config["test_batch_size"])
        else:
            ensemble.record_accuracies(i, train_loader_mini, test_loader_mini, numsamples_train=config['num_samples_train'], numsamples_val=config['num_samples_val'], val_attacks=config['val_attacks'], attack_iters=config['testing_attack_iters'], dataset_name=config['dataset_name'], restarts=config['testing_restarts'])
            print("ensemble accuracies:", ensemble.accuracies)

    return ensemble