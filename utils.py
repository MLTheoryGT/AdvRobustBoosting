# import apex.amp as amp
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

cifar10_mu_tup = (0.4914, 0.4822, 0.4465)
cifar10_std_tup = (0.2471, 0.2435, 0.2616)
cifar10_mu = torch.tensor(cifar10_mu_tup).view(3,1,1).cuda()
cifar10_std = torch.tensor(cifar10_std_tup).view(3,1,1).cuda()
cifar10_upper_limit = ((1 - cifar10_mu)/ cifar10_std)
cifar10_lower_limit = ((0 - cifar10_mu)/ cifar10_std)

cifar100_mu_tup = (0.507, 0.487, 0.441) 
cifar100_std_tup = (0.267, 0.256, 0.276)
cifar100_mu = torch.tensor(cifar100_mu_tup).view(3,1,1).cuda()
cifar100_std = torch.tensor(cifar100_std_tup).view(3,1,1).cuda()
cifar100_upper_limit = ((1 - cifar100_mu)/ cifar100_std)
cifar100_lower_limit = ((0 - cifar100_mu)/ cifar100_std)

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

"""
    Takes in a pytorch dataset object and returns train/test datasets after transformations
"""
def applyDSTrans(config):
    train_transforms = []
    test_transforms = []
    
    dataset = config["dataset"]
    if dataset == datasets.CIFAR10 or dataset==datasets.CIFAR100:
        for elt in [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]:
            train_transforms.append(elt)
    
#     print(dataset)
#     print(dict(dataset))
#     print(dataset==datasets.CIFAR10)
#     print("mean:", cifar10_mean)
#     print("std:", cifar10_std)
    
    tens = transforms.ToTensor()
    train_transforms.append(tens)
    test_transforms.append(tens)
    
    print(cifar10_mu_tup, cifar10_std_tup)
    
    if dataset == datasets.CIFAR10 and config["training_method"] != "trades" and not config.get('auto_attack', False):
        print("Normalized DS")
        norm = transforms.Normalize(cifar10_mu_tup, cifar10_std_tup)
        train_transforms.append(norm)
        test_transforms.append(norm)

    if dataset == datasets.CIFAR100 and config["training_method"] != "trades" and not config.get('auto_attack', False):
        print("Normalized DS")
        norm = transforms.Normalize(cifar100_mu_tup, cifar100_std_tup)
        train_transforms.append(norm)
        test_transforms.append(norm)
                
#     assert(len(train_transforms) == 4)
#     assert(len(test_transforms) == 2)

    train_ds = dataset('./data', train=True, download=True, transform=transforms.Compose(train_transforms))
    test_ds = dataset('./data', train=False, download=True, transform=transforms.Compose(test_transforms))

    return train_ds, test_ds

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    num_workers = 2
    train_dataset = datasets.CIFAR10(
        dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(
        dir_, train=False, transform=test_transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader

class SnapshotEnsembleScheduler:
    def __init__(self, opt, T, M, a0):
        self.opt = opt
        self.T = T
        self.M = M
        self.t = 0
        self.lastLR = a0
        self.a0 = a0

    def step(self):
        self.t+=1
        newLR = self.a0/2*(np.cos((np.pi*((self.t - 1) % np.ceil(self.T/self.M)))/ np.ceil(self.T/self.M)) + 1)

        for g in self.opt.param_groups:
            g['lr'] = newLR

        self.lastLR = newLR


    def get_last_lr(self):
        return [self.lastLR]

    def snapshot(self):
        return self.t % np.ceil(self.T/self.M) == 0