# -*- coding: utf-8 -*-

import time
import argparse
import os.path as osp
import pdb
import sys
import numpy as np
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# torch.backends.cudnn.enabled = False

# --------------------------------------------------------------------------------------------------------------
# Paths and modules
# --------------------------------------------------------------------------------------------------------------

biol_dir = "/hpcfs/home/da.martinez33/Biologia"

ACCpath = osp.join(biol_dir, 'results/nn/trainResults/mnist_net')
modelsPath = osp.join(biol_dir, 'results/nn/trainModels/mnist_net')

classifiers_folder = os.path.join(biol_dir, 'Codes', 'classification')
sys.path.append(classifiers_folder)

from load_data import loadDataBase
from mnist_net import Net as selected_net

class_labs = ['Fish', 'Insect']

# --------------------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='learning rate decay factor (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before '
                         'logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
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


def defineModel(Net=selected_net, load_net=False, model_file=''):
    """ Define model net, optimizer and loss criterion """
    global model
    global optimizer
    global criterion
    global load_model
    global res_flag

    model = Net()
    # ########################################
    #model.apply(weights_init)
    ##########################################33
    res_flag = 0
    if args.resume != '':  # For training from a previously saved state
        model.load_state_dict(torch.load(args.resume))
        res_flag = 1
    print(model)

    if args.cuda:
        model.cuda()

    load_model = load_net
    if osp.exists(args.save):
        with open(args.save, 'rb') as fp:
            state = torch.load(fp)
            model.load_state_dict(state)

    if load_model:
        if osp.exists(model_file):
            with open(model_file, 'rb') as fp:
                state = torch.load(fp)
                model.load_state_dict(state)

    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    criterion = nn.CrossEntropyLoss()  # nn.BCELoss().cuda() #nn.SoftMarginLoss()


def train(train_loader, epoch, k_fold=0, saveFolder=''):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.float()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        if target.size()[0] > 1:
            loss = criterion(output, torch.squeeze(target))
        else:
            loss = criterion(output, target[:,0])
        train_loss += loss.item()
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
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), train_loss / (batch_idx + 1.0),
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

    with open(osp.join(ACCpath, saveFolder, 'ACC_train_fold_{}.txt'.format(k_fold)), 'a') as f:
        f.write(line_to_save_train)
    print(line_to_save_train)


def test(test_loader, epoch, k_fold=0, saveFolder=''):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.float()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.float()
            output = model(data)
            if target.size()[0] > 1:
                test_loss += criterion(output, torch.squeeze(target)).item()
            else:
                test_loss += criterion(output, target[:,0]).item()
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
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAcc: {:.2f}%'.format(
                    epoch, (batch_idx + 1) * len(data), len(test_loader.dataset),
                           100. * (batch_idx + 1) / len(test_loader), test_loss / (batch_idx + 1.0), 
                           acccuracy_batch))
    test_loss = test_loss
    # loss function already averages over batch size
    test_loss /= len(test_loader)
    acccuracy = 100. * correct / len(test_loader.dataset)
    line_to_save_test = 'Test set: Average loss: {:.4f} Accuracy: {}/{} {:.0f}\n'.format(test_loss,
                                                                                         correct,
                                                                                         len(test_loader.dataset),
                                                                                         acccuracy)
    test_accs.append(acccuracy)
    saveDir = osp.join(osp.join(ACCpath, saveFolder))
    if not osp.isdir(saveDir):
        os.makedirs(saveDir)

    with open(osp.join(ACCpath, saveFolder, 'ACC_test_fold_{}.txt'.format(k_fold)), 'a') as f:
        f.write(line_to_save_test)
    print(line_to_save_test)

    return test_loss

def test_final(test_loader, epoch, best_model_file, which_net=selected_net):
    defineModel(Net=which_net, load_net=True, model_file=best_model_file)
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.float()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            #data = data.float()
            output = model(data)
            # get the index of the max log-probability
            out_probs = output.data.cpu().numpy()
            pred = output.data.max(1)[1]    
            pred2 = pred.cpu().numpy()
            if target.shape[0] > 1:
                targets = torch.squeeze(target)
            else:
                targets = target[:,0]
            targets = target.data.cpu().numpy()
            if batch_idx == 0:
                all_predictions = pred2
                all_probas = out_probs
                all_targets = targets
            else:
                all_predictions = np.append(all_predictions, pred2, axis=0)
                all_probas = np.append(all_probas, out_probs, axis=0)
                all_targets = np.append(all_targets, targets, axis=0)
            pred2 = np.expand_dims(pred2, axis=1)
            pred2 = torch.from_numpy(pred2)
            pred2 = pred2.long().cuda()
            pred = pred2
            correct += pred.eq(target.data).cpu().sum()

    accuracy = 100. * correct / len(test_loader.dataset)
    print("\nFinal accuracy: ", accuracy.item())

    return (all_targets, all_predictions, all_probas, accuracy)


def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(train_data, train_labels, test_data, test_labels, num_fold=0, Net_type='mnist_net', Ei=1):
    """Function to train all combinations for """
    print('Start training...\n')
    global test_accs
    test_accs = []

    print("\nFold number: {}\n".format(num_fold))

    #saveComb = file[:-4]  # Folder to save results and models
    saveComb = "{}_{}_{}_{}".format(args.batch_size, args.epochs, Net_type, args.lr) # Dependiendo de kwargs
    loaders = loadDataBase(args, train_data, train_labels, test_data, test_labels)

    if Net_type == 'mnist_net':
        from mnist_net import Net as s_net
    elif Net_type == 'mnist_net_2':
        from mnist_net_2 import Net as s_net
    elif Net_type == 'mnist_net_dropout':
        from mnist_net_dropout import Net as s_net
    else:
        from mnist_net import Net as s_net

    defineModel(Net=s_net)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    saveFolder = osp.join(args.outf, saveComb)

    try:
        for epoch in range(Ei, args.epochs + 1):
            epoch_start_time = time.time()
            train(loaders[0], epoch, k_fold=num_fold, saveFolder=saveComb)
            test_loss = test(loaders[1], epoch, k_fold=num_fold, saveFolder=saveComb)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:.2f}s ({:.2f}h)'.format(
                epoch,
                time.time() - epoch_start_time, 
                (time.time() - epoch_start_time) / 3600.0))
            print('-' * 89)
            
            scheduler.step()

            if args.save_model:
                if osp.isdir(saveFolder):
                    torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))
                else:
                    os.makedirs(saveFolder)
                    torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (saveFolder, epoch))
        ## Final test
        test_accs = np.asarray(test_accs)
        best_epoch = np.where(test_accs == np.amax(test_accs))[0][-1]
        best_model_file = '%s/model_epoch_%d.pth' % (saveFolder, best_epoch)
        train_results = test_final(loaders[1], epoch, best_model_file, which_net=s_net)

        final_results = [model, train_results[0], train_results[1], train_results[2], train_results[3]]
        return final_results


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
        sys.exit()



# if __name__ == '__main__':
#     best_loss = None
#     load_model = False
#     res_flag = 0
#     if load_model:
#         best_loss = test(0)
#     # GPS
#     if res_flag == 0:
#         Ei = 1
#     else:
#         if args.resume[-6] == '_':
#             Ei = int(args.resume[-5]) + 1
#             print('-' * 89)
#             print('Resuming from epoch %d' % (Ei))
#             print('-' * 89)
#         else:
#             Ei = int(args.resume[-6:-4]) + 1
#             print('-' * 89)
#             print('Resuming from epoch %d' % (Ei))
#             print('-' * 89)
#     # GPS
#     try:
#         main()

#     except KeyboardInterrupt:
#         print('-' * 89)
#         print('Exiting from training early')
#         sys.exit()
