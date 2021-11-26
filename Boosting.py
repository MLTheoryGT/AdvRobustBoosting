from BaseModels import BaseNeuralNet
from Architectures import PreActResNet18, WideResNet
from torchvision import datasets, transforms
from utils import applyDSTrans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from WongBasedTraining import WongBasedTrainingCIFAR10
from BaseModels import Validator
import torch.cuda as cutorch
import gc
import sys
from datetime import datetime
from Ensemble import Ensemble
from AdversarialAttacks import attack_fgsm, attack_pgd
import csv
import os
from autoattack.square import SquareAttack
from autoattack.autopgd_base import APGDAttack_targeted
from torch.utils.data import Subset
# from SquareAttack import SquareAttack

def SchapireWongMulticlassBoosting(config):
    print("attack_eps_wl: ", config['attack_eps_wl'])
    print("train_eps_wl: ", config['train_eps_wl'])
    if config["train_eps_wl"] == 0:
        print("Non adv training...")
    
    train_ds, test_ds = applyDSTrans(config)
    train_ds.targets = torch.tensor(np.array(train_ds.targets))
    test_ds.targets = torch.tensor(np.array(test_ds.targets))

    m = len(train_ds)
    k = len(train_ds.classes)
    
    # Regular loaders used for training
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size_wl'], shuffle=True)

    train_loader_default = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size_wl'], shuffle=False)
    test_loader_default = torch.utils.data.DataLoader(test_ds, batch_size=config['batch_size_wl'], shuffle=False)

    f = np.zeros((m, k))
    print(f)
    
    ensemble = Ensemble(weak_learner_type=config['weak_learner_type'], attack_eps=[], model_base=config['model_base'], weakLearners=[])
    
    start = datetime.now()
    
    path_head = f"./models/{config['training_method']}/{config['dataset_name']}/{config['num_samples_wl']}Eps{config['train_eps_wl']}/"
    print("path_head:", path_head)
    if os.path.exists(path_head):
        print("Already exists, exiting")
        return
    else:
        os.mkdir(path_head)
    
    for t in range(config['num_wl']):
        print("-"*100)
        print("Training weak learner {} of {}".format(t, config['num_wl']))
        # Boosting matrix update
        D_t = np.exp(f)
        D_t[np.arange(m), train_ds.targets] = 0
        
        # Set up boosting samplers
        train_sampler = BoostingSampler(train_ds, D_t)
        train_loader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler, batch_size=config['batch_size_wl'])
        
        # Fit WL on weighted subset of data
        h_i = config['weak_learner_type'](attack_eps=config['attack_eps_wl'], model_base=config['model_base'])
        
        h_i.fit(train_loader, test_loader, config)

        # Get training and test acuracy of WL
        _, predictions, _ = pytorch_predict(h_i.model, test_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_test_acc = (predictions == test_ds.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_val'].append(wl_test_acc)
        print("Test accuracy of weak learner: ", wl_test_acc)
        
        _, predictions, _ = pytorch_predict(h_i.model, train_loader_default, torch.device('cuda')) #y_true, y_pred, y_pred_prob
        wl_train_acc = (predictions == train_ds.targets.numpy()).astype(int).sum()/len(predictions)
        
        ensemble.accuracies['wl_train'].append(wl_train_acc)
        print("Training accuracy of weak learner: ", wl_train_acc)
        

        # update f one batch at a time via f[np.arange(batchsize)]
        advBatchSize = train_loader_default.batch_size
        a = 0
        advPredictions = np.zeros(m)
        
        # untargeted attacks
        
        for advCounter, data in enumerate(train_loader_default):
            X = data[0].cuda()
            y = data[1].cuda()
            if config['attack_name'] == 'apgd-t':
                apgd = APGDAttack_targeted(h_i.predict, n_restarts=1, n_iter=100, verbose=False,
                                          eps=0.127, norm='Linf', eot_iter=1, rho=.75, device='cuda')
#                 apgd = APGDAttack(h_i.predict, n_restarts=5, n_iter=100, verbose=False,
#                 eps=0.127, norm='Linf', eot_iter=1, rho=.75, device='cuda')
#                 apgd.loss = 't'
                x = X.clone().cuda()
                if len(x.shape) == 3:
                    x.unsqueeze_(dim=0)
                x_adv = apgd.perturb(X, y)
                predictions = h_i.predict(x_adv).argmax(axis=1)
                
            print("predictions shape:", predictions.shape)
            predictions = predictions.detach().int().cpu().numpy()
            upper_bound = min(len(train_loader_default.dataset), (advCounter + 1)*advBatchSize)
            advPredictions[np.arange(advCounter*advBatchSize, upper_bound)] = predictions
        
        # REMOVE THIS 
#         advPredictions = torch.clone(train_ds.targets).numpy()
#         advPredictions[:200] += 1
#         advPredictions[:200] %= k
#         perturb_indices = np.unique(np.random.randint(low=0, high=advPredictions.shape[0], size=7000))
#         print("perturb_indices shape:", perturb_indices.shape)
#         advPredictions[perturb_indices] += 1
#         advPredictions[perturb_indices] %= k
        
#         advPredictions = np.random.randint(low=0, high=k, size=m)
        
        print("After allindices: ", datetime.now()-start)
#         print("Predictions: ", advPredictions[:10]) 

        model_path = f'{path_head}wl_{t}.pth'
        torch.save(h_i.model.state_dict(), model_path)
        ensemble.addWeakLearner(model_path, 0)
        
        alpha = 0.01
        ensemble.updateWeakLearnerWeight(-1, alpha)
        
        print("Alpha:", alpha)
        
        y_train = train_ds.targets
        correctIndices = (advPredictions == y_train.numpy())
        incorrectIndices = (advPredictions != y_train.numpy())
        
        loss = -np.ones((m, k))
        correctLabels = y_train[correctIndices]
        loss[correctIndices,correctLabels] = 0
        y_hat = advPredictions[incorrectIndices]
        print("yhat", y_hat)
        y_hat = y_hat.astype(int)
        
        incorrectLabels = y_train[incorrectIndices].numpy()
        num_incorrect = incorrectLabels.shape[0]
        targeted_loss = np.zeros((num_incorrect,k))
        targeted_loss[:,y_hat] = 1
        # pick a class uniformly from [k] - {\hat{y}, y}
        rand_classes = np.random.randint(k-2, size=(num_incorrect))
#         print("rand_classes:", rand_classes.shape)
#         print("incorrect_labels:", incorrectLabels.shape)
#         print("y_hat:", y_hat.shape)
        rand_classes[rand_classes >= np.minimum(incorrectLabels, y_hat)] += 1
        rand_classes[rand_classes >= np.maximum(incorrectLabels, y_hat)] += 1
        
        # targeted attacks
        
        
#         incorrectDS = Subset(datasets.CIFAR10('./data', train=True, download=True, transform=None), incorrect_ds_indices)
        incorrectDS = Subset(train_ds, np.where(incorrectIndices==True)[0])
#         print("np.where: ", np.where(incorrectIndices==True))
#         print("np.where shape: ", np.where(incorrectIndices==True).shape)
        #         incorrectDS = train_ds[incorrectIndices]
# #         define trainloader
        train_loader = torch.utils.data.DataLoader(incorrectDS, batch_size=config['batch_size_wl'], shuffle=False)
        for i, (X, _) in enumerate(train_loader):
            X = X.cuda()
#             print("i: ", i)
            l = config['batch_size_wl'] * i
            r = min(l + config['batch_size_wl'], num_incorrect)
            length = r - l + 1
            y = torch.tensor(rand_classes[l:r]).cuda()
#             print("X:", X)
#             print("X shape:", X.shape)
            x_adv = attack_pgd(X, y, 0.127, h_i.predict, dataset_name=config['dataset_name'], targeted=True)
            y_pred_adv = h_i.predict(x_adv).argmax(dim=1).cpu().numpy()
            # y: targets
            # y_pred_adv: predictions
            # update loss wherever predictions = targets
            y = y.cpu().numpy()
            targeted_loss[l:r,y][y==y_pred_adv] = k - 2
        
        loss[incorrectIndices] = targeted_loss
        
        f += alpha * loss
        
        del h_i.model
        del h_i
        del predictions
        
        print("After WL ", t, " time elapsed(s): ", (datetime.now() - start).seconds)
        
    weight_path = f'{path_head}wl_weights.csv'
    print("weights:", ensemble.weakLearnerWeights)
    
    with open(weight_path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(ensemble.weakLearnerWeights)
        
    return ensemble


def pytorch_predict(model, test_loader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # set model to evaluate model
    model.eval()
    
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)
            
            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)
    
    y_true = y_true.cpu().numpy()  
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()
    
    return y_true, y_pred, y_pred_prob


from torch.utils.data.sampler import Sampler
class BoostingSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, C):
        self.indices = indices
        self.C = C

    def __iter__(self):
        sampleWeights = self.C.sum(axis = 1)
        probs = sampleWeights/sampleWeights.sum()
        maxNum = len(probs)
        chosen = np.random.choice(maxNum, size=maxNum, replace=True, p=probs)
        # print("Sampler chosen shape: ", chosen.shape)
        return np.nditer(chosen)

    def __len__(self):
        return len(self.indices)
      
    def setC(self, C):
        self.C = C

def runBoosting(config):

    from datetime import datetime

    t0 = datetime.now()
    
    ensemble = SchapireWongMulticlassBoosting(config)

    print("Finished in", (datetime.now()-t0).total_seconds(), " s")
    
    return ensemble
