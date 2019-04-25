from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
from pointnet_xyzs import PointNetCls
import torch.nn.functional as F
if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

test_result_numpy = np.empty(shape=(0, 5))
train_result_numpy = np.empty(shape=(0, 5))


try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(k = num_classes, num_points = opt.num_points)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, seg, target = data
        points, target = Variable(points), Variable(target[:,0])
        points = points.transpose(2,1) # size of [32, 3, 2500]
        seg = seg.unsqueeze_(-1) # size of [32, 2500, 1]
        seg = seg.transpose(2, 1) # size of [32, 1, 2500]
        seg = seg.float()
        points = torch.cat((points, seg), 1)

        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(),correct.item() / float(opt.batchSize)))

        current_train_result_numpy = np.array([[epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize)]])
        train_result_numpy = np.concatenate((train_result_numpy, current_train_result_numpy), axis = 0)

        pd.DataFrame(train_result_numpy).to_csv("log_xyzs_train.csv")

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, seg, target = data
            points, target = Variable(points), Variable(target[:,0])
            points = points.transpose(2,1)

            seg = seg.unsqueeze_(-1) # size of [32, 2500, 1]
            seg = seg.transpose(2, 1) # size of [32, 1, 2500]
            seg = seg.float()
            points = torch.cat((points, seg), 1)

            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

            current_test_result_numpy = np.array([[epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize)]])
            test_result_numpy = np.concatenate((test_result_numpy, current_test_result_numpy), axis = 0)
            pd.DataFrame(test_result_numpy).to_csv("log_xyzs_test.csv")

    torch.save(classifier.state_dict(), '%s/cls_model_xyzs_%d.pth' % (opt.outf, epoch))
