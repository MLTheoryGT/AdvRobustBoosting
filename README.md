# Adversarially Boosted Neural Networks
## Overview
Hi!  This repo contains work on adversarially boosted NNs.  The adversarial boosting method is inspired by ADABOOST.MM in [Mukherjee et al. 2013](https://www.cs.princeton.edu/~schapire/papers/multiboost.pdf).  This repo contains code to train ensembles of NNs as well as test their adversarial accuracy against PGD attacks.  The method used to train the individual NNs can be changed, but primarily we use the method given in [Wong et al](https://arxiv.org/abs/2001.03994)

## Installation

Required libraries:
- libraries in requirements.txt 

```
pip install -r requirements.txt
```

- Apex instructions (based on https://github.com/NVIDIA/apex):
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
```

- Autoattack
```
pip install git+https://github.com/fra31/auto-attack
```

## File Descriptions

- BaseModels.py: Includes the architecture of Wong et al. and abstract classes for training / validation
- WideResNet.py: Includes the architecture for WideResNets
- WongBasedTraining.py: Wong method for training a weak learner
- Ensemble.py: class for an ensemble of weak learners
- Boosting.py: Creates an ensemble from the weaklearners (based on [Mukherjee et al. 2013](https://www.cs.princeton.edu/~schapire/papers/multiboost.pdf))
- CIFARWeakLearner.ipynb: Includes code cells for training / validating a weak learner
- CIFARBoosting.ipynb: Includes code cells for training / validating an ensemble
- Testing.py Includes code for validation of an ensemble
- `./Models` is where the individual weak learners of an ensemble are saved.  For memory purposes we don't simply load all weak learners of an ensemble into memory at once
- `./results` contains all the plots from MNIST and CIFAR10Experiments

## How to run an experiment from scratch

### Training the ensemble
Run the Train Ensemble Section of `CifarBoosting.py`

Parameters to set:

Training parameters: maxSamples, epsilons, train_eps_nn, num_wl, model_base, batch_size, attack_iters, restarts

Validation parameters:
attack_iters, restarts, val_attacks, numsamples_train, numsamples_val

### Testing the Ensemble
Run the Test Ensemble Section of `CifarBoosting.py`
