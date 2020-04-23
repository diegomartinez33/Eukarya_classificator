# -*- coding: utf-8 -*-
import numpy as np
import time
import itertools
from itertools import cycle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

plt.rcParams.update({'font.size': 12})

class_labs = ['W','1','2','3','4','Rem','Art']

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_conf_matrix(y_test, y_pred, savePath='', class_names=class_labs):
    """ Function to obtain and save confusion matrix"""
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    fig1 = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    fig1.savefig(savePath, bbox_inches = 'tight')
    plt.close(fig1)
    # Plot normalized confusion matrix
    fig2 = plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    fig2.savefig(savePath[:-4] + 'normalized' + '.png', bbox_inches = 'tight')
    plt.close(fig2)

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
                                                         average="macro")
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
        from itertools import cycle
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

def f_score(y_test, y_pred, averaged='Yes', class_names=class_labs):
    """ Function to obtain the f score depending on
         labels and probabilities predicted from the model"""
    if averaged == 'Yes':
        values = precision_recall_fscore_support(y_test, y_pred, average='macro')
        return values[2]
    else:
        values = classification_report(y_test, y_pred, target_names=class_names)
        return values

def ACC_score(y_true, y_pred):
    """ Function to obtain the ACC value from predicted
         labels and groundtruth labels """
    acc = accuracy_score(y_true, y_pred)
    return acc

