# -*- coding: utf-8 -*-

import time
import argparse
import os.path as osp
import pdb
import sys

import torch
import torch.optim as optim
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import datasets, transforms

# GPS
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# torch.backends.cudnn.enabled = False

# --------------------------------------------------------------------------------------------------------------
# Paths and modules
# --------------------------------------------------------------------------------------------------------------

TesisPath = '/hpcfs/home/da.martinez33/Tesis'

ACCpath = osp.join(TesisPath, 'results/trainResults', 'Adam')
modelsPath = osp.join(TesisPath, 'results/trainModels', 'Adam')

data_base_path = osp.join(TesisPath, 'Data', 'DOE_databases', 'balancedData')
data_base_file = 'dataBaseDict_SC_amp_30_3.pkl'

codesTrainfolder = osp.join(TesisPath, 'Codes', 'Train_pytorch')
netsfolder = osp.join(codesTrainfolder, 'nets')

sys.path.append(codesTrainfolder)

from load_data import loadDataBase
from test_metrics import testMetrics
from test_metrics_2 import testMetrics

sys.path.append(netsfolder)

from out_artifacts_net import Net_30s
from out_artifacts_net import Net_10s
from out_artifacts_net import Net_30s_ST
from out_artifacts_net import Net_10s_ST

class_labs = ['W','1','2','3','4','Rem']

# --------------------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=0.0005, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save', type=str, default='model.pt',
                    help='file on which to save model weights')
parser.add_argument('--outf', default=modelsPath, help='folder to output images and model checkpoints')
parser.add_argument('--resume', default='', help="path to model (to continue training)")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


# --------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------

def trainDOEexp(balancedPath=data_base_path,Ei=1):
    """Function to train all combinations for """
    print('Start training\n')
    for file in os.listdir(balancedPath):
        if file.endswith(".pkl") and file == data_base_file:
            print(file)
            saveComb = file[:-4]  # Folder to save results and models
            for i in range(1,4):
                loaders = loadDataBase(args, balancedPath, file, trainFold=i, elim_artif=True)
                windowSize = int(file[-8:-6])
                dataType = file[-15:-13]
                if windowSize == 30:
                    if dataType == 'SC':
                        defineModel(Net=Net_30s)
                    else:
                        defineModel(Net=Net_30s_ST)
                elif windowSize == 10:
                    if dataType == 'SC':
                        defineModel(Net=Net_10s)
                    else:
                        defineModel(Net=Net_10s_ST)

                try:
                    for epoch in range(Ei, args.epochs + 1):
                        epoch_start_time = time.time()
                        train(loaders[0],epoch, saveFolder=saveComb, trainFold=i)
                        test_loss = test(loaders[1] ,epoch, saveFolder=saveComb, trainFold=i)
                        print('-' * 89)
                        print('| end of epoch {:3d} | time: {:5.2f}s ({:.2f}h)'.format(
                            epoch, time.time() - epoch_start_time, (time.time() - epoch_start_time) / 3600.0))
                        print('-' * 89)
                        saveFolder = osp.join(args.outf, saveComb, str(i))

                        if osp.isdir(saveFolder):
                            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))  # GPS
                        else:
                            os.makedirs(saveFolder)
                            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))
                    testMetrics(loaders[2], model, args, criterion, ACCpath, saveFolder=saveComb, trainFold=i,
                                classes=class_labs)

                except KeyboardInterrupt:
                    print('-' * 89)
                    print('Exiting from training early')
                    sys.exit()


def weights_init(m):
    """ custom weights initialization called on netG and netD """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.00, 0.01)
        # m.bias.data.normal_(0.00, 0.1)
        m.bias.data.fill_(0.1)
        # xavier(m.weight.data)
        # xavier(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        # m.weight.data.normal_(1.0, 0.01)
        # m.bias.data.fill_(0)


def defineModel(Net=Net_30s):
    """ Define model net, optimizer and loss criterion """
    global model
    global optimizer
    global criterion
    global load_model
    global res_flag

    model = Net()
    # GPS
    model.apply(weights_init)
    res_flag = 0
    if args.resume != '':  # For training from a previously saved state
        model.load_state_dict(torch.load(args.resume))
        res_flag = 1
    print(model)
    # GPS

    if args.cuda:
        model.cuda()

    load_model = False
    if osp.exists(args.save):
        with open(args.save, 'rb') as fp:
            state = torch.load(fp)
            model.load_state_dict(state)
            load_model = True

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    criterion = nn.CrossEntropyLoss()  # nn.BCELoss().cuda() #nn.SoftMarginLoss()


def train(train_loader, epoch,saveFolder='',trainFold=1):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, torch.squeeze(target))
        train_loss += loss.data[0]
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        pred2 = np.expand_dims(pred2, axis=1)
        pred2 = torch.from_numpy(pred2)
        pred2 = pred2.long().cuda()
        pred = pred2
        # ---------------------------------------------
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data) * (batch_idx + 1.0))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} ({:.3f})\tAcc: {:.2f}% '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0], train_loss / (batch_idx + 1.0),
                acccuracy_batch))

    train_loss = train_loss
    # loss function already averages over batch size
    train_loss /= len(train_loader)
    acccuracy = 100. * correct / len(train_loader.dataset)
    line_to_save_train = 'Train set: Average loss: {:.4f} Accuracy: {}/{} {:.0f}\n'.format(train_loss,
                                                                                           correct,
                                                                                           len(train_loader.dataset),
                                                                                           acccuracy)

    saveDir = osp.join(osp.join(ACCpath, saveFolder))
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)

    with open(osp.join(ACCpath, saveFolder, 'ACC_train' + str(trainFold) + '.txt'), 'a') as f:
        f.write(line_to_save_train)
    print(line_to_save_train)


def test(val_loader, epoch,saveFolder='',trainFold=1):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        pred2 = np.expand_dims(pred2, axis=1)
        pred2 = torch.from_numpy(pred2)
        pred2 = pred2.long().cuda()
        pred = pred2
        correct += pred.eq(target.data).cpu().sum()
        acccuracy_batch = 100. * correct / (len(data) * (batch_idx + 1.0))
        if batch_idx % args.log_interval == 0:
            print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(val_loader.dataset),
                       100. * batch_idx / len(val_loader), test_loss / (batch_idx + 1.0), acccuracy_batch))
    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(val_loader)
    acccuracy = 100. * correct / len(val_loader.dataset)
    line_to_save_test = 'Test set: Average loss: {:.4f} Accuracy: {}/{} {:.0f}\n'.format(test_loss,
                                                                                         correct,
                                                                                         len(val_loader.dataset),
                                                                                         acccuracy)
    saveDir = osp.join(osp.join(ACCpath, saveFolder))
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)

    with open(osp.join(ACCpath, saveFolder, 'ACC_test' + str(trainFold) + '.txt'), 'a') as f:
        f.write(line_to_save_test)
    print(line_to_save_test)

    return test_loss


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



if __name__ == '__main__':
    best_loss = None
    load_model = False
    res_flag = 0
    if load_model:
        best_loss = test(0)
    # GPS
    if res_flag == 0:
        Ei = 1
    else:
        if args.resume[-6] == '_':
            Ei = int(args.resume[-5]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
        else:
            Ei = int(args.resume[-6:-4]) + 1
            print('-' * 89)
            print('Resuming from epoch %d' % (Ei))
            print('-' * 89)
    # GPS
    try:
        trainDOEexp()

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.exit()
