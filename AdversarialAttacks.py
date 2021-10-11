import torch
import torch.nn.functional as F
from utils import (cifar10_std, cifar10_upper_limit, cifar10_lower_limit)
from utils import (cifar100_std, cifar100_upper_limit, cifar100_lower_limit)
import matplotlib.pyplot as plt

# taken from https://github.com/locuslab/fast_adversarial/blob/master/MNIST/evaluate_mnist.py
def attack_fgsm(X, y, epsilon, model):
    delta = torch.zeros_like(X, requires_grad=True)
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    delta.data = epsilon * torch.sign(grad)
    del output
    del loss
    ans = delta.detach()
    del delta
    return ans


def attack_pgd_mnist(X, y, epsilon, model, alpha, attack_iters=5, restarts=1):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon).cuda()
        delta.data = clamp(delta, 0-X, 1-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)[0]
            if len(index) == 0:
                break
#             loss = F.cross_entropy(output, y)
            loss = F.nll_loss(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            d = clamp(d, 0-X, 1-X)
            delta.data[index] = d[index]
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_pgd(X, y, epsilon, model, attack_iters=20, restarts=1, dataset_name='cifar10', change_eps=True):
#     print("pgd called with", epsilon, alpha, attack_iters, restarts)
#     alpha = alpha * 100
#     print(alpha)
#     print("x30")
    if dataset_name == 'cifar10':
        (std, upper_limit, lower_limit) = (cifar10_std, cifar10_upper_limit, cifar10_lower_limit)
    elif dataset_name == 'cifar100':
        (std, upper_limit, lower_limit) = (cifar100_std, cifar100_upper_limit, cifar100_lower_limit)
    alpha = (2/255.)/std
    
    if change_eps:
        epsilon = torch.tensor([[[epsilon]], [[epsilon]], [[epsilon]]]).cuda()
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for attackIter in range(attack_iters):
#             print("attackIter: ", attackIter)
            output = model(X + delta)
#             print(output[0:10])
#             output = output * 30
#             output = torch.nn.functional.softmax(output)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        
    
    flatDelta = torch.flatten(max_delta).squeeze().cpu().numpy()
# uncomment if i want delta to be printed out
#     print(flatDelta)
#     plt.hist(flatDelta)
#     plt.show()
    return max_delta

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)