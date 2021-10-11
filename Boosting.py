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
    
    ensemble = Ensemble(weak_learner_type=config['weak_learner_type'], attack_eps=[], model_base=config['model_base'], weakLearners=[])
    
    start = datetime.now()
    
    path_head = f"./models/{config['training_method']}/{config['dataset_name']}/{config['num_samples_wl']}Eps{config['train_eps_wl']}/"
    print("path_head:", path_head)
    if os.path.exists(path_head):
        print("Already exists, exiting")
        return
    else:
        os.mkdir(path_head)
    
    def gcLoop():
        print("-"*100)
        print("Training weak learner {} of {}".format(t, config['num_wl']))
        # Boosting matrix update
 
        C_t = np.zeros((m, k))
        fcorrect = f[np.arange(m), train_ds.targets]
        fexp = np.exp(f - fcorrect[:,None])
        C_t = fexp.copy()
        fexp[np.arange(m), train_ds.targets] = 0
        C_t[np.arange(m), train_ds.targets] = -np.sum(fexp, axis=1)
        C_tp = np.abs(C_t)
        
        # Set up boosting samplers
        train_sampler = BoostingSampler(train_ds, C_tp)
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
        allIndices = np.zeros(m)
        for advCounter, data in enumerate(train_loader_default):
            X = data[0].cuda()
            y = data[1].cuda()
            # Maybe rewrite the below in a cleaner fashion?
            is_specific = ('attack_name' in config and config['attack_name'])
            
            if config["train_eps_wl"] == 0:
                delta = 0
            elif not is_specific: # CHANGE
                delta = attack_pgd(X, y, config['attack_eps_wl'][0], h_i.predict, restarts=1, attack_iters=20)
#             delta = attack_fgsm(X, y, config['attack_eps_wl'][0], h_i.predict)
            if 'attack_name' not in config or not config['attack_name']:
                predictions = h_i.predict(X + delta).argmax(axis=1)
            elif config['attack_name'] == 'square':
                # Use eps=[0.127] since data is normalized
                square = SquareAttack(h_i.predict, p_init=.8, n_queries=100, eps=0.127, norm='Linf',
                n_restarts=1, seed=0, verbose=False, device=torch.device('cuda'), resc_schedule=False)
                print("X shape:", X.shape)
                x = X.clone().cuda()
                if len(x.shape) == 3:
                    x.unsqueeze_(dim=0)
                print("x shape:", x.shape)
                x_adv = square.perturb(X, y)
                predictions = h_i.predict(x_adv).argmax(axis=1)
            elif config['attack_name'] == 'apgd-t':
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
            indices = predictions.detach().int().cpu().numpy()
            print("indices shape:", indices.shape)
            upper_bound = min(len(train_loader_default.dataset), (advCounter + 1)*advBatchSize)
            allIndices[np.arange(advCounter*advBatchSize, upper_bound)] = indices
        
        print("After allindices: ", datetime.now()-start)
        print("Predictions: ", allIndices[:10])
        
        # Get alpha for this weak learners
        a = -C_t[np.arange(m), allIndices.astype(int)].sum()
        b = fexp.sum()
        delta_t = a / b
        alpha = 1/2*np.log((1+delta_t)/(1-delta_t))
        print("Alpha: ", alpha)
        print("before pessimistic update: ", datetime.now()-start)
        
        y_train = train_ds.targets
        correctIndices = (allIndices == y_train.numpy())
        incorrectIndices = (allIndices != y_train.numpy())
        f[np.arange(m), y_train] += alpha * correctIndices
        f[incorrectIndices, :] += alpha
        f[np.arange(m), y_train] -= alpha * incorrectIndices
    
        # save WL model and alpha
        model_path = f'{path_head}wl_{t}.pth'
        torch.save(h_i.model.state_dict(), model_path)
        
        del h_i.model
        del h_i
        del predictions
        
        torch.cuda.empty_cache()
        ensemble.addWeakLearner(model_path, alpha)
        print("t: ", t, "memory allocated:", cutorch.memory_allocated(0))   
        print("After WL ", t, " time elapsed(s): ", (datetime.now() - start).seconds)
    
        
    #end of gc loop
    for t in range(config['num_wl']):
        gcLoop()
        gc.collect()
        
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
