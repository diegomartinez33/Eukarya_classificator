
import numpy as np
import os
import os.path as osp
import sys

import torch
import torch.utils.data as data_utils
from random import shuffle

biol_dir = "/hpcfs/home/da.martinez33/Biologia"

def loadDataBase(args, train_data, train_labels, test_data, test_labels):
    """ Function to load data as a tensor dataloader for training in Pytorch"""

    data_train = train_data
    label_train = train_labels
    data_test = test_data
    label_test = test_labels

    data_train = np.expand_dims(np.expand_dims(data_train, axis=1), axis=1)
    data_test = np.expand_dims(np.expand_dims(data_test, axis=1), axis=1)

    label_train = np.expand_dims(label_train, axis=1)
    label_train = label_train.astype(np.int64)

    label_test = np.expand_dims(label_test, axis=1)
    label_test = label_test.astype(np.int64)

    print(data_train.shape)
    print(label_train.shape)
    print(data_test.shape)
    print(label_test.shape)

    # adjust train data
    traindata = torch.from_numpy(data_train)
    trainlabel = torch.from_numpy(label_train)
    train = data_utils.TensorDataset(traindata, trainlabel)
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

    # adjust test data
    testdata = torch.from_numpy(data_test)
    testlabel = torch.from_numpy(label_test)
    test = data_utils.TensorDataset(testdata, testlabel)
    test_loader = data_utils.DataLoader(test, batch_size=args.test_batch_size, shuffle=True)

    finalTest = {}
    finalTest['data'] = testdata
    finalTest['labels'] = testlabel

    loaders = (train_loader,test_loader,finalTest)
    return loaders
