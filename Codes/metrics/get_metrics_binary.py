# -*- coding: utf-8 -*-
import numpy as np
import time
#import itertools
#from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#from sklearn.preprocessing import label_binarize

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from itertools import cycle

plt.rcParams.update({'font.size': 12})

class_labs = ['fishes', 'insects']


def ACC_score(train_cv_list):
    """ Function to obtain the ACC value from predicted
         labels and groundtruth labels """
    folds = len(train_cv_list)
    for i in range(folds):
        y_true, y_pred = train_cv_list[i][1], train_cv_list[i][2]
        acc = accuracy_score(y_true, y_pred)
        print('ACC for fold {0}: {1:0.2f}'.format(i, acc))

def classif_report(train_cv_list, class_names=class_labs):
    folds = len(train_cv_list)
    for i in range(folds):
        y_true, y_pred = train_cv_list[i][1], train_cv_list[i][2]
        class_rep = classification_report(y_true, y_pred, target_names=class_names)
        print('Classification report for fold {}'.format(i))
        print(class_rep)

def get_prf(train_cv_list):
    folds = len(train_cv_list)
    for i in range(folds):
        y_true, y_pred = train_cv_list[i][1], train_cv_list[i][2]
        results = precision_recall_fscore_support(y_true, y_pred, pos_label=1)
        print('P-R and F1 for fold {}'.format(i))
        print('Precision: {}'.format(results[0]))
        print('Recall: {}'.format(results[1]))
        print('F1_score: {}'.format(results[2]))

def mcc(train_cv_list):
    folds = len(train_cv_list)
    for i in range(folds):
        y_true, y_pred = train_cv_list[i][1], train_cv_list[i][2]
        mcc = matthews_corrcoef(y_true, y_pred)
        print('MCC value for fold {} is: {}'.format(i, mcc))

def p_r_curve(y_test, probas_pred, averaged='Yes', savePath='',class_names=class_labs):
    """ Function to create the precision-recall curve depending on
     labels and probabilities predicted from the model"""
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = len(class_names)
    y_test = label_binarize(y_test, list(range(0, n_classes)))

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            probas_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], probas_pred[:, i])
    
    print('y_test shape: ', y_test.shape)
    print('probas_pred shape: ', probas_pred.shape)
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                    probas_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, probas_pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # ---- Plot micro-averaged P-R curve
    if averaged == 'Yes':
        fig2=plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))
        plt.close(fig2)
        fig2.savefig(savePath, bbox_inches = 'tight')
    # ---- Plot P-R curve for each class
    else:
        # setup plot details
        colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        plt.figure(figsize=(7, 8))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.2f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(n_classes), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                          ''.format(i, average_precision[i]))

        fig3 = plt.gcf()
        fig3.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Extension of Precision-Recall curve to multi-class')
        plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=12))

        fig3.savefig(savePath, bbox_inches = 'tight')
        plt.close(fig3)

def p_r_curve_cv(train_cv_list, savePath=''):
    # setup plot details
    precision = dict()
    recall = dict()
    average_precision = dict()
    folds = len(train_cv_list)
    #print(train_cv_list)
    #y_test = label_binarize(y_test, list(range(0, fold_num)))

    for i in range(folds):
        y_test, probas_pred = train_cv_list[i][1], train_cv_list[i][2]
        precision[i], recall[i], _ = precision_recall_curve(y_test,
                                                            probas_pred)
        average_precision[i] = average_precision_score(y_test, probas_pred)

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for i, color in zip(range(folds), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for fold {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig3 = plt.gcf()
    fig3.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves for different folds')
    plt.legend(lines, labels, loc=(0, -.45), prop=dict(size=12))

    fig3.savefig(savePath, bbox_inches = 'tight')
    plt.close(fig3)

def ROC_curve(train_cv_list, savePath=''):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    folds = len(train_cv_list)
    for i in range(folds):
        y_test, y_score = train_cv_list[i][1], train_cv_list[i][3]
        fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:,0], pos_label=1)
        roc_auc[i] = roc_auc_score(y_test, y_score[:,0])

    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

    plt.figure(figsize=(7, 8))
    #f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []

    for i, color in zip(range(folds), colors):
        l, = plt.plot(fpr[i], tpr[i], color=color, lw=2)
        lines.append(l)
        labels.append('ROC curve for fold {0} (area = {1:0.2f})'
                      ''.format(i, roc_auc[i]))

    fig3 = plt.gcf()
    lw = 2
    fig3.subplots_adjust(bottom=0.25)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for different folds')
    plt.legend(lines, labels, loc="lower right", prop=dict(size=12))

    fig3.savefig(savePath, bbox_inches = 'tight')
    plt.close(fig3)



