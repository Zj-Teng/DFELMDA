# -*- coding: utf-8 -*-

# 循环10次
import numpy as np
from matplotlib import pyplot
from numpy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import scipy.io
from model import AEMDA
from utils import calculate_performace



if __name__ == "__main__":
    probafs = []
    probascore = []

    ae_y_pred_probafs = []
    ae_y_pred_probascore = []
    all_performance = []

    t = 0
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(10):
        probaresult, ae_y_pred_probresult, labels = AEMDA()
        probafs.extend(probaresult)
        ae_y_pred_probafs.extend(ae_y_pred_probresult)

    probafs = np.array(probafs)
    ae_y_pred_probafs = np.array(ae_y_pred_probafs)

    b1 = probafs.reshape((50, 37917))
    b2 = ae_y_pred_probafs.reshape((50, 37917))

    for i in range(5):
        test_label = np.array([x for j, x in enumerate(labels) if j % 5 == i])
        real_labels = []
        for val in test_label:
            if val[0] == 1:  # tuples in array, val[0]- first element of tuple
                real_labels.append(0)
            else:
                real_labels.append(1)

        probascore = sum(b1[i::5])/5
        ae_y_pred_probascore = sum(b2[i::5])/5

        acc, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(real_labels), probascore, real_labels)
        fpr, tpr, auc_thresholds = roc_curve(real_labels, ae_y_pred_probascore)
        auc_score = auc(fpr, tpr)
        scipy.io.savemat('raw_DNN', {'fpr': fpr, 'tpr': tpr, 'auc_score': auc_score})

        precision1, recall, pr_threshods = precision_recall_curve(real_labels, ae_y_pred_probascore)
        aupr_score = auc(recall, precision1)
        print(acc, precision, sensitivity, MCC, auc_score, aupr_score, f1_score)
        all_performance.append([acc, precision, sensitivity, MCC, auc_score, aupr_score, f1_score])

        print(acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score)
        all_performance.append([acc, precision, sensitivity, specificity, MCC, auc_score, aupr_score, f1_score])
        t = t + 1

        pyplot.plot(fpr, tpr, label='ROC fold %d (AUC = %0.4f)' % (t, auc_score))
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('ROC Curves')
        pyplot.legend()

    mean_tpr /= 5
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pyplot.plot(mean_fpr, mean_tpr, '--', linewidth=2.5, label='Mean ROC (AUC = %0.4f)' % mean_auc)

    pyplot.show()
    print('mean performance')
    print(np.mean(np.array(all_performance), axis=0))
    Mean_Result = np.mean(np.array(all_performance), axis=0)
    print('---' * 20)
    print('Mean-Acc=', Mean_Result[0], '\n Mean-pre=', Mean_Result[1])
    print('Mean-Sen=', Mean_Result[2], '\n Mean-Spe=', Mean_Result[3])
    print('Mean-MCC=', Mean_Result[4], '\n' 'Mean-auc=', Mean_Result[5])
    print('Mean-Aupr=', Mean_Result[6], '\n' 'Mean_F1=', Mean_Result[7])