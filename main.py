import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
from datetime import datetime
import numpy as np

import datasets
import models
from lib.non_parametric_classifier import NonParametricClassifier
from lib.graph_structure import GraphStructure
from lib.criterion import Criterion, UELoss
from lib.protocols import kNN
from lib.utils import AverageMeter
from lib.normalize import Normalize
from lib.LinearAverage import LinearAverage
# from lib.visualization import tSNE
from matplotlib import pyplot as plt


def config():
    global args
    parser = argparse.ArgumentParser(description='config for super-AND')

    parser.add_argument('--dataset', default='cifar10', type=str, help='available dataset: cifar10, cifar100 (dafault: cifar10)')
    parser.add_argument('--network', default='resnet18', type=str, help='available network: resnet18, resnet101 (default: resnet18)')

    parser.add_argument('--structure', default='BFS', type=str)

    parser.add_argument('--npc_t', default=0.1, type=float, metavar='T', help='temperature parameter for softmax')
    parser.add_argument('--npc_m', default=0.5, type=float, help='momentum for non-parametric updates')

    parser.add_argument('--low_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--neighbor_size', default=4, type=int)
    parser.add_argument('--lr', default=0.03, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--epochs', default=200, type=int, help='max epoch per round. (default: 200)')
    parser.add_argument('--rounds', default=10, type=int, help='max iteration, including initialisation one. ''(default: 5)')

    parser.add_argument('--batch_size', default=128, type=int, metavar='B', help='training batch size')

    parser.add_argument('--model_dir', default='checkpoint/', type=str, help='model save path')
    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='test only')

    parser.add_argument('--seed', default=1567010775, type=int, help='random seed')

    args = parser.parse_args()
    return args


def preprocess(args):
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4866, 0.4409)
        std = (0.2009, 0.1984, 0.2023)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10SAND(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR10SAND(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)
    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100SAND(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
        testset = datasets.CIFAR100SAND(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, drop_last=True)
        
    return trainset, trainloader, testset, testloader


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 80:
        lr = args.lr * (0.1 ** ((epoch - 80) // 40))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(round, epoch, trainloader, net, npc, structure, criterion, optimizer, device):
    # tracking variables
    train_loss = AverageMeter()

    # switch the model to train mode
    net.train()

    # adjust learning rate
    adjust_learning_rate(optimizer, epoch)  
    optimizer.zero_grad()

    for batch_idx, (inputs, _, _, indexes) in enumerate(trainloader):
        inputs, indexes = inputs.to(device), indexes.to(device)

        features = net(inputs)
        neighbor_indexes = structure.neighbor_indexes_sim[indexes]
        outputs = npc(features, indexes, neighbor_indexes, round)

        loss = criterion(outputs, indexes, structure)
        loss.backward()
        train_loss.update(loss.item(), inputs.size(0))

        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 80 == 0:
            print('Round: {round} Epoch: [{epoch}][{elps_iters}/{tot_iters}] '
                  'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '.format(
                  round=round, epoch=epoch, elps_iters=batch_idx,
                  tot_iters=len(trainloader), train_loss=train_loss))


def main():
    args = config()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, trainloader, testset, testloader = preprocess(args)
    ntrain = len(trainset)

    net = models.__dict__['ResNet18withSobel'](low_dim=args.low_dim)
    npc = NonParametricClassifier(args.structure, args.low_dim, ntrain, args.npc_t, args.npc_m, args.device)
    # structure = GraphStructure(args.structure, ntrain, args.low_dim, args.batch_size, args.neighbor_size, args.device)
    criterion = Criterion()
    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.to(args.device)
    npc.to(args.device)
    # structure.to(args.device)
    criterion.to(args.device)

    print('==> init by self loop..')
    checkpoint = torch.load('checkpoint/init.t7')
    net.load_state_dict(checkpoint['net'])
    npc.load_state_dict(checkpoint['npc'])

    # images = trainset.data
    # neighbor_indexes_sim = checkpoint['structure']['neighbor_indexes_sim']
    # neighbor_indexes_disim = checkpoint['structure']['neighbor_indexes_disim']
    # query = torch.randperm(neighbor_indexes_sim.size(0))[:10]
    # for q in query:
    #     os.mkdir('BFS_bi/%d' % q)
    #     for top, i in enumerate(neighbor_indexes_sim[q]):
    #         img = images[i]
    #         plt.imshow(img)
    #         plt.savefig('BFS_bi/%d/%d' % (q,top))
    #     for top, i in enumerate(neighbor_indexes_disim[q]):
    #         img = images[i]
    #         plt.imshow(img)
    #         plt.savefig('BFS_bi/%d/%d_neg' % (q,top))
    # sys.exit(0)


    if len(args.resume) > 0:
        model_path = args.model_dir + args.resume
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.model_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        npc.load_state_dict(checkpoint['npc'])

    if args.test_only:
        acc = kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device=args.device)
        print("accuracy: %.2f\n" % (acc*100))  
        sys.exit(0)

    best_acc = 0
    cur_acc = []
    # for r in range(args.rounds):
    for r in range(1, args.rounds):
    # for r in range(1, 2):
        if r > 0:
            structure = GraphStructure(args.structure, ntrain, args.low_dim, args.batch_size, args.neighbor_size, args.device)
            structure.to(args.device)
            structure.update(npc)

        # for epoch in range(args.epochs):
        for epoch in range(1):
            train(r, epoch, trainloader, net, npc, structure, criterion, optimizer, args.device)
            acc = kNN(net, npc, trainloader, testloader, K=200, sigma=0.1, recompute_memory=False, device=args.device)
            print("accuracy: %.2f" % (acc*100))  

            if acc > best_acc:
                print("state saving...")
                state = {
                    'net': net.state_dict(),
                    'npc': npc.state_dict(),
                    'structure' : structure.state_dict(),
                    'acc': acc,
                    'round': r,
                    'epoch': epoch
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/{}_cur_reset.t7'.format(args.structure))
                best_acc = acc
            print("best accuracy: %.2f\n" % (best_acc*100))
        cur_acc.append(acc)
        print(cur_acc)
    sys.exit(0)


if __name__ == "__main__":
    main()
