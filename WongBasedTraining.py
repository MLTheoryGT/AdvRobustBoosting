from BaseModels import BaseNeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
cuda = torch.device('cuda:0')
import time
# import logging
from apex import amp
# from torch.cuda import amp
from utils import clamp
from AdversarialAttacks import attack_pgd
import copy
from utils import SnapshotEnsembleScheduler
from utils import (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)



class WongBasedTrainingCIFAR10(BaseNeuralNet):
    def __init__(self, model_base, attack_eps):
        super().__init__(model_base, attack_eps)
    
    def fit(self, train_loader, test_loader, config):
        if config['seed_wl'] is not None:
            np.random.seed(config["seed_wl"])
            torch.manual_seed(seed=config["seed_wl"])
            torch.cuda.manual_seed(seed=config["seed_wl"])
    
        if config['dataset_name'] == "cifar10":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar10_std_tup, cifar10_mu_tup, cifar10_std, cifar10_mu, cifar10_upper_limit, cifar10_lower_limit)
        elif config['dataset_name'] == "cifar100":
            (std_tup, mu_tup, std, mu, upper_limit, lower_limit) = (cifar100_std_tup, cifar100_mu_tup, cifar100_std, cifar100_mu, cifar100_upper_limit, cifar100_lower_limit)

#         train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
#          have our own loaders

        val_X = None
        val_y = None
        for data in test_loader:
            val_X = data[0].to(cuda)
            val_y = data[1].to(cuda)
            break

        epsilon = (config['train_eps_wl'] / 255.) / std
        alpha = (config['train_alpha_wl'] / 255.) / std
        pgd_alpha = (2 / 255.) / std

        model = self.model
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=(config['lr_max_wl'] if config['lr_schedule_wl'] != 'snapshot' else config['snapshot_a0_wl']), momentum=config['momentum_wl'], weight_decay=config['weight_decay_wl'])
        amp_args = dict(opt_level=config['opt_level_wl'], loss_scale=config['loss_scale_wl'], verbosity=False)
        if config['opt_level_wl'] == 'O2':
            amp_args['master_weights'] = True
#             unclear if True is what we want here
        model, opt = amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()

        if config['delta_init_wl'] == 'previous':
            delta = torch.zeros(train_loader.batch_size, 3, 32, 32).cuda()
        
        epoch_size = len(train_loader.dataset)
        num_epochs = max(1, config['num_samples_wl'] // epoch_size)
        lr_steps = num_epochs * len(train_loader)
        
        if config['lr_schedule_wl'] == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=config['lr_min_wl'], max_lr=config['lr_max_wl'],
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif config['lr_schedule_wl'] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
        elif config['lr_schedule_wl'] == 'snapshot':
            scheduler = SnapshotEnsembleScheduler(opt, lr_steps, config["snapshot_cycles_wl"], config["snapshot_a0_wl"])

        # Training
        prev_robust_acc = 0.
        start_train_time = time.time()
        print('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
        currSamples = 0 # added
        snapshots = 0
        for epoch in range(num_epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            print("train_loader:", train_loader) #delete
            for i, data in enumerate(train_loader):
#                 print("lr:", scheduler.get_last_lr())
                X, y = data[0], data[1]
                currSamples += train_loader.batch_size # added
                X, y = X.cuda(), y.cuda()
                if i == 0:
                    first_batch = (X, y)
                if config['delta_init_wl'] != 'previous':
                    delta = torch.zeros_like(X).cuda()
                if config['delta_init_wl'] == 'random':
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                if i % 100 == 99:
                    model.eval()
                    self.record_accuracies(currSamples, val_X=val_X, val_y=val_y, train_X=X, train_y=y, attack_iters=config["attack_iters_val_wl"], 
                                            restarts=config["restarts_wl"], val_attacks=config["val_attacks"], dataset_name=config["dataset_name"])
                    model.train()
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()

            if config['early_stop_wl']:
                # Check current PGD robustness of model using random minibatch
                X, y = first_batch
                pgd_delta = attack_pgd(X, y, epsilon, model, 5, 1, config['dataset_name'], change_eps=False)
                with torch.no_grad():
                    output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
                robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
                if robust_acc - prev_robust_acc < -0.2:
                    break
                prev_robust_acc = robust_acc
                best_state_dict = copy.deepcopy(model.state_dict())

                if config["lr_schedule_wl"] == "snapshot" and scheduler.snapshot():
                    print("doing snapshot, lr = ", scheduler.get_last_lr()[0])
                    torch.save(model.state_dict(), config["save_dir"] + str(config["snapshot_cycles_wl"] - snapshots - 1) + '.pth')
                    snapshots+=1
#             if early_stop:
#                 # Check current PGD robustness of model using random minibatch
#                 X, y = first_batch
#                 pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
#                 with torch.no_grad():
#                     output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
#                 robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
#                 if robust_acc - prev_robust_acc < -0.2:
#                     break
#                 prev_robust_acc = robust_acc
#                 best_state_dict = copy.deepcopy(model.state_dict())
            epoch_time = time.time()
            lr = scheduler.get_last_lr()[0]
            print('%d \t %.1f \t \t %.4f \t %.4f \t %.4f'%(epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n))
        
        train_time = time.time()
        print('Total train time: %.4f minutes', (train_time - start_train_time)/60)
    
    

