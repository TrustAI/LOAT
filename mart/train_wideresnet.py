import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('/home/xiangyuy/capacity_robustness')
from wideresnet import *
from resnet import *
from utils.train_utils import *
from utils.test_utils import *
from mart import mart_loss
import numpy as np
import time
import csv

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=7e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='wideresnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--adv', type=bool, default=True)
parser.add_argument('--backbone', type=str, default='wrn3410')
parser.add_argument('--reg', action='store_true', default=False)
parser.add_argument('--reg_type', type=str, default='mse')
parser.add_argument('--from_epoch', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.05)
parser.add_argument('--LORE_type', type=str, default='LORE_v1')
args = parser.parse_args()

model_dir = args.model

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda:1" if use_cuda else "cpu")
kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../../dataset_folder', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10)
testset = torchvision.datasets.CIFAR10(root='../../dataset_folder', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = mart_loss(model=model,
                           epoch=epoch,
                           x_natural=data,
                           y=target,
                           reg=args.reg,
                           reg_type=args.reg_type,
                           optimizer=optimizer,
                           device = device,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta,
                           gamma=args.gamma,
                           lot=args.LORE_type)
        loss.backward()
        optimizer.step()

        # print progress
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def train_std(args, model, device, train_loader, optimizer, epoch):

    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=20,
                  step_size=0.003):

    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err_pgd

def eval_adv_test_whitebox(model, device, test_loader):

    model.eval()
    robust_err_total_fgsm = 0
    robust_err_total_pgd7 = 0
    robust_err_total_pgd20 = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out = model(X)
        natural_err_total += (out.data.max(1)[1] != y.data).float().sum()
        
        err_robust_fgsm = _pgd_whitebox(model, X, y, num_steps=1, step_size=args.epsilon)
        err_robust_pgd7 = _pgd_whitebox(model, X, y, num_steps=7)
        err_robust_pgd20 = _pgd_whitebox(model, X, y, num_steps=20)

        robust_err_total_fgsm += err_robust_fgsm
        robust_err_total_pgd7 += err_robust_pgd7
        robust_err_total_pgd20 += err_robust_pgd20

    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total_fgsm / len(test_loader.dataset), 1- robust_err_total_pgd7 / len(test_loader.dataset), 1- robust_err_total_pgd20 / len(test_loader.dataset)

def eval_adv_train_whitebox(model, device, train_loader):

    model.eval()
    natural_err_total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out = model(X)
        natural_err_total += (out.data.max(1)[1] != y.data).float().sum()

    return 1 - natural_err_total / len(train_loader.dataset)


def eval_std(model, device, loader):
    model.eval()
    natural_err_total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out = model(X)
        err_natural = (out.data.max(1)[1] != y.data).float().sum()
        natural_err_total += err_natural

    print('natural_acc: ', 1 - natural_err_total / len(train_loader.dataset))
    return 1 - natural_err_total / len(loader.dataset)

def main():

    model = WideResNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   
    
    start_epoch = 1
        
    train_natural_acc = []
    train_robust_acc = []
    test_natural_acc = []
    test_robust_acc = []
    
    dataset = 'cifar10'
    
    csv_save_folder = '/home/xiangyuy/capacity_robustness/checkpoints_new/mart/'+dataset+'/csvs'
    model_save_folder = '/home/xiangyuy/capacity_robustness/checkpoints_new/mart/'+dataset+'/models'
    
    make_dirs(model_save_folder)
    make_dirs(csv_save_folder)

    if not args.adv:
        train_csv_path = os.path.join(csv_save_folder, 'Train-'+args.backbone+'-'+str(args.epochs)+'.csv')
        train_file = open(train_csv_path, 'w')
        train_csv_writer = csv.writer(train_file, delimiter=' ')
        train_csv_writer.writerow(['Epoch', 'Accuracy_Nat'])

        test_csv_path = os.path.join(csv_save_folder, 'Test-'+args.backbone+'-'+str(args.epochs)+'.csv')        
        test_file = open(test_csv_path, 'w')
        test_csv_writer = csv.writer(test_file, delimiter=' ')
        test_csv_writer.writerow(['Epoch', 'Accuracy_Nat'])
    else:
        if not args.reg:
            train_csv_path = os.path.join(csv_save_folder, 'Train-'+args.backbone+'-'+str(args.beta)+'-'+str(args.epochs)+'.csv')
            train_file = open(train_csv_path, 'w')
            train_csv_writer = csv.writer(train_file, delimiter=' ')
            train_csv_writer.writerow(['Epoch', 'Accuracy_Nat'])

            test_csv_path = os.path.join(csv_save_folder, 'Test-'+args.backbone+'-'+str(args.beta)+'-'+str(args.epochs)+'.csv')        
            test_file = open(test_csv_path, 'w')
            test_csv_writer = csv.writer(test_file, delimiter=' ')
            test_csv_writer.writerow(['Epoch', 'Accuracy_Nat', 'Accuracy_fgsm', 'Accuracy_pgd7', 'Accuracy_pgd20'])
        else:
            train_csv_path = os.path.join(csv_save_folder, 'Train-'+args.backbone+'-'+str(args.beta)+'-'+str(args.epochs)+'-reg-'+str(args.reg_type)+'-'+str(args.gamma)+'-'+args.LORE_type+'.csv')
            train_file = open(train_csv_path, 'w')
            train_csv_writer = csv.writer(train_file, delimiter=' ')
            train_csv_writer.writerow(['Epoch', 'Accuracy_Nat'])

            test_csv_path = os.path.join(csv_save_folder, 'Test-'+args.backbone+'-'+str(args.beta)+'-'+str(args.epochs)+'-reg-'+str(args.reg_type)+'-'+str(args.gamma)+'-'+args.LORE_type+'.csv')        
            test_file = open(test_csv_path, 'w')
            test_csv_writer = csv.writer(test_file, delimiter=' ')
            test_csv_writer.writerow(['Epoch', 'Accuracy_Nat',  'Accuracy_fgsm', 'Accuracy_pgd7', 'Accuracy_pgd20'])
        
    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        start_time = time.time()
        
        if args.adv:
            # adversarial training
            train(args, model, device, train_loader, optimizer, epoch)
        else:
            train_std(args, model, device, train_loader, optimizer, epoch)
        
        end_time = time.time()
        gap_time = end_time - start_time

        print('================================================================')
        
        if args.adv:
            train_natural_accuracy_total = eval_adv_train_whitebox(model, device, train_loader)
            test_natural_accuracy_total, test_robust_fgsm, test_robust_pgd7, test_robust_pgd20 = eval_adv_test_whitebox(model, device, test_loader)

            train_csv_writer.writerow([epoch, train_natural_accuracy_total.cpu().numpy()])
            test_csv_writer.writerow([epoch, test_natural_accuracy_total.cpu().numpy(), test_robust_fgsm.cpu().numpy(), test_robust_pgd7.cpu().numpy(), test_robust_pgd20.cpu().numpy(), gap_time])
        
        else:
            train_natural_accuracy_total = eval_std(model, device, train_loader)
            test_natural_accuracy_total = eval_std(model, device, test_loader)

            train_csv_writer.writerow([epoch, train_natural_accuracy_total.cpu().numpy()])
            test_csv_writer.writerow([epoch, test_natural_accuracy_total.cpu().numpy()])        
        
        print('================================================================')        
                     
        
        '''
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))
        '''

if __name__ == '__main__':
    main()