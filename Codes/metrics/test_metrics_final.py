
import torch
from torch.autograd import Variable

import numpy as np
import os.path as osp
import sys
import pickle

TesisPath = '/hpcfs/home/da.martinez33/de_marr/Tesis'

acc_path = osp.join(TesisPath, 'results/trainResults_raw')

codesTrainfolder = osp.join(TesisPath, 'Codes', 'Train_pytorch')
sys.path.append(codesTrainfolder)

import get_metrics_multilab as getmet
class_labs = ['W','1','2','3','4','Rem','Art']


def testMetrics( test_loader, model, args, criterion, ACCpath=acc_path, saveFolder='',trainFold=1,
                 classes=class_labs):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data = data.float()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, torch.squeeze(target)).data[0]
        # get the index of the max log-probability
        out_probs = output.data.cpu().numpy()
        pred = output.data.max(1)[1]
        pred2 = pred.cpu().numpy()
        target2 = target.data.cpu().numpy()
        if 'finalPred' in locals():
            finalPred = np.append(finalPred, pred2)

        else:
            finalPred = pred2

        if 'finalTarget' in locals():
            finalTarget = np.append(finalTarget, target2)
        else:
            finalTarget = target2

        if 'finalprobs' in locals():
            finalprobs = np.append(finalprobs, out_probs, axis=0)
        else:
            finalprobs = out_probs
    
    finalResults = {}
    finalResults['finalTarget'] = finalTarget
    finalResults['finalPred'] = finalPred
    finalResults['finalprobs'] = finalprobs
    
    savePathFile = osp.join(ACCpath, saveFolder, 'finalResults_' + str(trainFold) + '.pkl')
    pickle.dump(finalResults, open(savePathFile, 'wb'), pickle.HIGHEST_PROTOCOL)
    print('Saved dataBase in: ', savePathFile)

    # ----- Confusion matrix ---------
    confmatPath = osp.join(ACCpath, saveFolder, 'ConfMat_' + str(trainFold) + '.png')
    getmet.get_conf_matrix(finalTarget, finalPred, savePath=confmatPath, class_names=classes)
    print('Confusion Matrix created for: \t' + confmatPath)

    # ----- Precision-Recall Curve ---
    prCurvePath_aver = osp.join(ACCpath, saveFolder, 'PR_curve_averaged_' + str(trainFold) + '.png')
    getmet.p_r_curve(finalTarget, finalprobs, averaged='Yes', savePath=prCurvePath_aver, class_names=classes)
    prCurvePath_perClass = osp.join(ACCpath, saveFolder, 'PR_curve_classes_' + str(trainFold) + '.png')
    getmet.p_r_curve(finalTarget, finalprobs, averaged='No', savePath=prCurvePath_perClass, class_names=classes)
    print('Precision-Recall curve created for: \t' + prCurvePath_aver)
    print('Precision-Recall curve created for: \t' + prCurvePath_perClass)

    # ---- F-score -----
    fscore = getmet.f_score(finalTarget, finalPred, averaged='Yes', class_names=classes)
    f_score_path = osp.join(ACCpath, saveFolder, 'F_score_' + str(trainFold) + '.npy')
    f_score_path_text = osp.join(ACCpath, saveFolder, 'F_score_' + str(trainFold) + '.txt')
    np.save(f_score_path, fscore)
    with open(f_score_path_text, 'w') as f:
        f.write(str(fscore))
    print('F-score for: ' + f_score_path + 'is: ' + str(fscore))

    ClassReport = getmet.f_score(finalTarget, finalPred, averaged='No', class_names=classes)
    ClassReport_path = osp.join(ACCpath, saveFolder, 'Class_Report_' + str(trainFold) + '.txt')

    with open(ClassReport_path, 'w') as f:
        f.write(ClassReport)
    print('Classification Report: \n\n' + ClassReport)

    # ---- ACC score ----
    acc_score = getmet.ACC_score(finalTarget, finalPred)
    acc_score_path = osp.join(ACCpath, saveFolder, 'ACC_score_' + str(trainFold) + '.npy')
    acc_score_path_text = osp.join(ACCpath, saveFolder, 'ACC_score_' + str(trainFold) + '.txt')
    np.save(acc_score_path, acc_score)
    with open(acc_score_path_text, 'w') as f:
        f.write(str(acc_score))
    print('ACC-score for: ' + acc_score_path + 'is: ' + str(acc_score))
