from BaseModels import BaseNeuralNet
from Architectures import PreActResNet18, PreActResNet18_100, WideResNet
from torchvision import datasets, transforms
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
from AdversarialAttacks import attack_pgd
from utils import (cifar10_mu, cifar10_std, cifar100_mu, cifar100_std)

class Ensemble(Validator):
    def __init__(self, weak_learner_type, attack_eps, model_base, weakLearners=[], weakLearnerWeights=[]):
        """
        """
        Validator.__init__(self, 'Number of weak learners', attack_eps)
        self.weakLearners = weakLearners
        self.weakLearnerWeights = weakLearnerWeights
        self.weakLearerWeightsTensor = torch.tensor(weakLearnerWeights, requires_grad=False).unsqueeze(1).float().cuda()
        self.weak_learner_type = weak_learner_type
        self.accuracies['wl_train'] = []
        self.accuracies['wl_val'] = []
        self.attack_eps = attack_eps
        self.model_base = model_base
        if model_base == PreActResNet18:
            self.num_classes = 10
        elif model_base == PreActResNet18_100:
            self.num_classes = 100
        
        assert len(self.weakLearners) == 0

    def plot_wl_acc(self, path=None):
        plt.subplots()
        plt.plot(self.train_checkpoints, self.accuracies['wl_train'])
        plt.plot(self.train_checkpoints, self.accuracies['wl_val'])
        plt.legend(['Train accuracy', 'Val accuracy'])
        plt.xlabel(self.xlabel)
        plt.ylabel('Weak learner accuracy')
        plt.title('Weak learner accuracy')
        plt.grid()
        if path is not None:
            plt.savefig(path, dpi=250)
        plt.show()
    
    def addWeakLearner(self, weakLearner, weakLearnerWeight):
        self.weakLearners.append(weakLearner)
        self.weakLearnerWeights.append(weakLearnerWeight)
        self.weakLearnerWeightsTensor = torch.tensor(self.weakLearnerWeights, requires_grad=False).unsqueeze(1).float().cuda()
    
    def updateWeakLearnerWeight(self, idx, weakLearnerWeight):
        self.weakLearnerWeights[idx] = weakLearnerWeight
        self.weakLearnerWeightsTensor = torch.tensor(self.weakLearnerWeights, requires_grad=False).unsqueeze(1).float().cuda()
    
    def getSingleWLPrediction(self, i, wLPredictions, X):
#         print("memory allocated:", cutorch.memory_allocated(0), "for WL num ", i)
        if not isinstance(self.weakLearners[i], str):
            learner = self.weakLearners[i]
        else:
            learner = self.weak_learner_type(model_base=self.model_base, attack_eps=self.attack_eps)
#             print("learner model:", learner.model)
#             print("model base:", self.model_base)
            learner.model.load_state_dict(torch.load(self.weakLearners[i]))
            learner.model = learner.model.to(torch.device('cuda:0'))
            learner.model.eval()
        prediction = learner.model(X)
        wLPredictions[:,:,i] = F.softmax(prediction, dim=1)
#         wLPredictions[:,:,i] = prediction
            
            
    def getWLPredictions(self, X, k):
        T = len(self.weakLearners)
#         print("total num weak learners:", T)
#         print("Start WL Pred memory allocated:", cutorch.memory_allocated(0)) 
        wLPredictions = torch.zeros((X.shape[0], k, T)).cuda()

                    
        for i in range(T):
            self.getSingleWLPrediction(i, wLPredictions, X)
        
        torch.cuda.empty_cache()
        
#         print("Finishing WL pred memory allocated:", cutorch.memory_allocated(0))
        return wLPredictions

    def predict(self, X):
        # We hope to deprecate schapirePredict since we shouldn't need anymore
#         learner = self.weak_learner_type()
#         learner.model.load_state_dict(torch.load(self.weakLearners[0]))
#         learner.model = learner.model.to(torch.device('cuda:0'))
#         learner.model.eval()
#         prediction = learner.model(X)
#         print("calling modded predict")
#         return prediction
    
        return self.schapireContinuousPredict(X, self.num_classes)

    def predictUnnormalizedDataCIFAR10(self, X):
        X_norm = (X - cifar10_mu) / cifar10_std
        return self.schapireContinuousPredict(X_norm, self.num_classes)

    def predictUnnormalizedDataCIFAR100(self, X):
        X_norm = (X - cifar100_mu) / cifar100_std
        return self.schapireContinuousPredict(X_norm, self.num_classes)


    def schapirePredict(self, X, k):
        print("shouldn't be here")
        wLPredictions = None

        predictions = np.zeros(len(X))
        T = len(self.weakLearners)
        
        wLPredictions = self.getWLPredictions(X, k).argmax(axis=1).transpose(1, 0)

        for i in range(len(X)):
            F_Tx =[]
            for l in range(k):
                F_Tx.append(sum([self.weakLearnerWeights[t] * (1 if wLPredictions[t][i] == l else 0) for t in range(T)]))
        
            predictions[i] = np.argmax(np.array(F_Tx))
        return predictions
    
    def schapireContinuousPredict(self, X, k):
        wLPredictions = None

        T = len(self.weakLearners)
        
        wLPredictions = self.getWLPredictions(X, k)
#         print(wlPre)
        weights = self.weakLearnerWeightsTensor
        assert(wLPredictions.shape == (len(X), k, T))
#         print("weights shape:", weights.shape)
#         print("T:", T)
        assert(weights.shape == (T, 1))
        output = torch.matmul(wLPredictions, weights).squeeze(2)
        del wLPredictions
        assert(output.shape == (len(X), k))
        
        output = F.normalize(output, p=1, dim=1)
        output = torch.log(output)
        return output
    
    def gradOptWeights(self, train_loader, num_samples=1000, train_eps=0.127):
        # gradient optimize weights
        # get ~100 adv examples
        # calculate cross-entropy loss
        # calc gradient with respect to weights
        print("weights before opt:", self.weakLearnerWeights)
        print("tensor before opt:", self.weakLearnerWeightsTensor)
        self.toggleWeightGrad(True)
        optim = torch.optim.Adam([self.weakLearnerWeightsTensor], lr=0.01)
        total_samples = 0
        for _, data in enumerate(train_loader):
            X, y = data[0].cuda(), data[1].cuda()
            X_adv = attack_pgd(X, y, train_eps, self.predict)
            if total_samples > num_samples:
                break
            total_samples += len(X)
            output = self.predict(X_adv)
            optim.zero_grad()
            loss = F.cross_entropy(output, y)
            loss.backward()
            optim.step()
        self.toggleWeightGrad(False)
        self.weakLearnerWeights = self.weakLearnerWeightsTensor.squeeze(1).tolist()
        print("weights after opt:", self.weakLearnerWeights)
        print("tensor after opt:", self.weakLearnerWeightsTensor)    
    
    def toggleWeightGrad(self, option=True):
        self.weakLearnerWeightsTensor.requires_grad = option

    def get_sum(self, dicts):
        ans = {}
        for d in dicts:
            for k in d:
                if isinstance(k, str) and k == 'train' or k == 'val':
                    if k not in ans:
                        ans[k] = d[k]
                    else:
                        ans[k] += d[k]
                else:
                    if k not in ans:
                        ans[k] = np.array(d[k])
                    else:
                        ans[k] += np.array(d[k])
        return ans
    
    def get_mean(self, dicts):
        ans = {}
        for d in dicts:
            for k in d:
                if isinstance(k, str) and k == 'train' or k == 'val':
                    if k not in ans:
                        ans[k] = d[k]
                    else:
                        ans[k] += d[k]
                else:
                    if k not in ans:
                        ans[k] = np.array(d[k])
                    else:
                        ans[k] += np.array(d[k])
        for k in ans:
            ans[k] = ans[k] / len(dicts)
        return ans

    def record_accuracies(self, progress, train_loader, test_loader, numsamples_train, numsamples_val, val_attacks, attack_iters, restarts, dataset_name):
        # record train
        if numsamples_train >0:
            train_batch_size = train_loader.batch_size
            self.train_checkpoints.append(progress)
            # sum losses
            # average accuracies
            curSample = 0
            train_loss_dicts = []
            train_acc_dicts = []
            for i, data in enumerate(train_loader):
                curSample += train_batch_size
                if curSample >= numsamples_train: break
                losses, accuracies = self.calc_accuracies(data[0].cuda(), data[1].cuda(), data_type='train', dataset_name=dataset_name)
                train_loss_dicts.append(losses)
                train_acc_dicts.append(accuracies)
            self.losses['train'].append(self.get_sum(train_loss_dicts)['train'])
            self.accuracies['train'].append(self.get_mean(train_acc_dicts)['train'])
        
        # record val / adversarial
        
        val_batch_size = test_loader.batch_size
        self.val_checkpoints.append(progress)
        curSample = 0
        val_loss_dicts = []
        val_acc_dicts = []
        for i, data in enumerate(test_loader):
            curSample += val_batch_size
            if curSample >= numsamples_val: break
            losses, accuracies = self.calc_accuracies(data[0].cuda(), data[1].cuda(), data_type='val', val_attacks=val_attacks, attack_iters=attack_iters, restarts=restarts, dataset_name=dataset_name)
            val_loss_dicts.append(losses)
            val_acc_dicts.append(accuracies)
            print(accuracies)
        val_loss_dict = self.get_sum(val_loss_dicts)
        val_acc_dict = self.get_mean(val_acc_dicts)
        self.losses['val'].append(val_loss_dict['val'])
        self.accuracies['val'].append(val_acc_dict['val'])
        for k in val_loss_dict:
            if isinstance(k, str): continue
            attack = k
            if len(self.losses[attack.__name__]) == 0:
                self.losses[attack.__name__] = [[] for i in range(len(self.attack_eps))]
                self.accuracies[attack.__name__] = [[] for i in range(len(self.attack_eps))]
            for i in range(len(self.attack_eps)):
                self.losses[attack.__name__][i].append(val_loss_dict[attack][i])
                self.accuracies[attack.__name__][i].append(val_acc_dict[attack][i])
        