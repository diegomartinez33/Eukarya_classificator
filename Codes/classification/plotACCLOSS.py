#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:08:56 2017

@author: oem
"""

import numpy as np
# import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

tesisPath = '/hpcfs/home/da.martinez33/Biologia'
results_path = os.path.join(tesisPath, 'results', 'nn','trainResults', 'mnist_net','10000_30')

num_epochs=30

plt.rcParams.update({'font.size': 14})

def countlines(fold):
    pat = fold
    acc_file = os.path.join(results_path, "ACC_test_fold_" + pat + ".txt")
    count = len(open(acc_file).readlines())
    print("\nPaciente {} Lineas: {}".format(pat,count))
    return count

def makeplot(fold):
    lines = countlines(fold)
    epoch = np.linspace(1, num_epochs, num=num_epochs)
    pat = fold
    train_file = os.path.join(results_path, "ACC_train_fold_" + pat + ".txt")
    test_file = os.path.join(results_path, "ACC_test_fold_" + pat + ".txt")
    
    # Get info from txt files
    train1 = np.genfromtxt(train_file)
    loss1 = train1[lines-num_epochs:lines,4]
    acc1 = train1[lines-num_epochs:lines,7]
    
    test1 = np.genfromtxt(test_file)
    loss_te1 = test1[lines-num_epochs:lines,4]
    acc_te1 = test1[lines-num_epochs:lines,7]
    
    
#    train2 = np.genfromtxt(base_path + '/ACC_train2.txt')
#    loss2 = train2[:num_epochs,4]
#    acc2 = train2[:num_epochs,7]
#    
#    test2 = np.genfromtxt(base_path + '/ACC_test2.txt')
#    loss_te2 = test2[:num_epochs,4]
#    acc_te2 = test2[:num_epochs,7]
#    
#    train3 = np.genfromtxt(base_path + '/ACC_train3.txt')
#    loss3 = train3[:num_epochs,4]
#    acc3 = train3[:num_epochs,7]
#    
#    test3 = np.genfromtxt(base_path + '/ACC_test3.txt')
#    loss_te3 = test3[:num_epochs,4]
#    acc_te3 = test3[:num_epochs,7]
    
    # ---------------- Plot ACC and Loss info per Fold ----------------------
    fig1=plt.figure()
    plt.plot(epoch, acc1, 'c', label='ACC_train',color="blue")
    plt.plot(epoch, acc_te1, 'c', label='ACC_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy for fold ' + pat)
    plt.legend()
    
    print('Figure 1 ploted')
    
    fig2=plt.figure()
    plt.plot(epoch, loss1, 'c', label = 'Loss_train',color="blue")
    plt.plot(epoch, loss_te1, 'c', label = 'Loss_test',color="red")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss for fold ' + pat)
    plt.legend()
    
    fig1.savefig(results_path + '/' + 'Fold_ACC_fold_{}'.format(fold) + '.png')
    fig2.savefig(results_path + '/' + 'Fold_Loss_fold_{}'.format(fold) + 'png')
    
    print('Figure 2 ploted')
    
    plt.close(fig1)
    plt.close(fig2)
    
    
#    fig1=plt.figure()
#    plt.plot(epoch, acc2, 'c', label='ACC2_train',color="blue")
#    plt.plot(epoch, acc_te2, 'c', label='ACC2_test',color="red")
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.title('Training and Validation Accuracy for Fold 2')
#    plt.legend()
#    
#    print('Figure 3 ploted')
#    
#    fig2=plt.figure()
#    plt.plot(epoch, loss2, 'c', label = 'Loss2_train',color="blue")
#    plt.plot(epoch, loss_te2, 'c', label = 'Loss2_test',color="red")
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.title('Training and Validation Loss for Fold 2')
#    plt.legend()
#    
#    fig1.savefig(base_path + '/' + 'Fold2_ACC.png')
#    fig2.savefig(base_path + '/' + 'Fold2_Loss.png')
#    
#    print('Figure 4 ploted')
#    
#    plt.close(fig1)
#    plt.close(fig2)
#    
#    fig1=plt.figure()
#    plt.plot(epoch, acc3, 'c', label='ACC3_train',color="blue")
#    plt.plot(epoch, acc_te3, 'c', label='ACC3_test',color="red")
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.title('Training and Validation Accuracy for Fold 3')
#    plt.legend()
#    
#    print('Figure 5 ploted')
#    
#    fig2=plt.figure()
#    plt.plot(epoch, loss3, 'c', label = 'Loss3_train',color="blue")
#    plt.plot(epoch, loss_te3, 'c', label = 'Loss3_test',color="red")
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.title('Training and Validation Loss for Fold 3')
#    plt.legend()
#    
#    fig1.savefig(base_path + '/' + 'Fold3_ACC.png')
#    fig2.savefig(base_path + '/' + 'Fold3_Loss.png')
#    
#    print('Figure 6 ploted')
#    
#    plt.close(fig1)
#    plt.close(fig2)
#    
#    # ---------------- Plot ACC and Loss info for all folds -------------------
#    print('Start plots of all Folds')
#    
#    fig1 = plt.figure()
#    plt.plot(epoch,acc1,color="blue",label='Fold 1')
#    plt.plot(epoch,acc2,color="red",label='Fold 2')
#    plt.plot(epoch,acc3,color="green",label='Fold 3')
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.title("Training Accuracy for Three Folds")
#    plt.legend()
#    
#    print('Figure 1 ploted')
#    
#    fig2 = plt.figure()
#    plt.plot(epoch,acc_te1,color="blue",label='Fold 1')
#    plt.plot(epoch,acc_te2,color="red",label='Fold 2')
#    plt.plot(epoch,acc_te3,color="green",label='Fold 3')
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.title("Validation Accuracy for Three Folds")
#    plt.legend()
#    
#    print('Figure 2 ploted')
#    
#    fig3 = plt.figure()
#    plt.plot(epoch,loss1,color="blue",label='Fold 1')
#    plt.plot(epoch,loss2,color="red",label='Fold 2')
#    plt.plot(epoch,loss3,color="green",label='Fold 3')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.title("Training Loss for Three Folds")
#    plt.legend()
#    
#    print('Figure 3 ploted')
#    
#    fig4 = plt.figure()
#    plt.plot(epoch,loss_te1,color="blue",label='Fold 1')
#    plt.plot(epoch,loss_te2,color="red",label='Fold 2')
#    plt.plot(epoch,loss_te3,color="green",label='Fold 3')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')  
#    plt.title("Validation Loss for Three Folds")
#    plt.legend()
#    
#    print('Figure 4 ploted')
#    
#    fig1.savefig(base_path + '/' + 'ACC_Train_3Folds.png', bbox_inches = 'tight')
#    fig2.savefig(base_path + '/' + 'ACC_val_3Folds.png', bbox_inches = 'tight')
#    fig3.savefig(base_path + '/' + 'Loss_Train_3Folds.png')
#    fig4.savefig(base_path + '/' + 'Loss_val_3Folds.png')
#    
#    plt.close(fig1)
#    plt.close(fig2)
#    plt.close(fig3)
#    plt.close(fig4)
    
# Create all plots of experiments combinations
# testFolder = 'dataBaseDict_ST_amp_30_3'

# makeplot(testFolder)

# sys.exit()
if os.path.isdir(results_path):
    for file in os.listdir(results_path):
        fold = str(file[-5:-4])
        makeplot(fold)
        #countlines(folder)



