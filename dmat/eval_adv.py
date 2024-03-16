"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.attacks import create_attack
from core.data import get_data_info
from core.data import load_data
from core.models import create_model
from core.metrics import accuracy

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils.context import ctx_noparamgrad_and_eval



# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + '/' + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'
elif args.data in ['tiny-imagenet', 'tiny-imagenets']:
    da = '/tiny-imagenet/'


DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/state-last.pt'

log_path = LOG_DIR + '/log-adv.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
# BATCH_SIZE = args.batch_size
# BATCH_SIZE_VALIDATION = args.batch_size_validation
BATCH_SIZE = 128
BATCH_SIZE_VALIDATION = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    logger.log('Evaluating on testing set.')
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)

print(args.model)
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint

std_acc = 0.0
        
for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    out = model(x)
    std_acc += accuracy(y, out)

std_acc /= len(test_dataloader)
logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(std_acc*100))


criterion = nn.CrossEntropyLoss()

# Evaluation on Linf-PGD attack with iteration-10

eval_attack_linf_pgd_10 = create_attack(model, criterion, args.attack, args.attack_eps, 10, args.attack_step)

adv_acc = 0.0

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = eval_attack_linf_pgd_10.perturb(x, y)            
        out = model(x_adv)
        adv_acc += accuracy(y, out)
adv_acc /= len(test_dataloader)
logger.log('Linf-PGD-10 Accuracy-\tTest: {:.2f}%.'.format(adv_acc*100))

# Evaluation on Linf-PGD attack with iteration-20

eval_attack_linf_pgd_20 = create_attack(model, criterion, args.attack, args.attack_eps, 20, args.attack_step)

adv_acc = 0.0

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = eval_attack_linf_pgd_20.perturb(x, y)            
        out = model(x_adv)
        adv_acc += accuracy(y, out)
adv_acc /= len(test_dataloader)
logger.log('Linf-PGD-20 Accuracy-\tTest: {:.2f}%.'.format(adv_acc*100))

# Evaluation on Linf-PGD attack with iteration-40

eval_attack_linf_pgd_40 = create_attack(model, criterion, args.attack, args.attack_eps, 40, args.attack_step)

adv_acc = 0.0

for x, y in test_dataloader:
    x, y = x.to(device), y.to(device)
    with ctx_noparamgrad_and_eval(model):
        x_adv, _ = eval_attack_linf_pgd_40.perturb(x, y)            
        out = model(x_adv)
        adv_acc += accuracy(y, out)
adv_acc /= len(test_dataloader)
logger.log('Linf-PGD-40 Accuracy-\tTest: {:.2f}%.'.format(adv_acc*100))

# Evaluation on Auto attack with apgd-ce and apgd-t

norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION)

print ('Script Completed.')