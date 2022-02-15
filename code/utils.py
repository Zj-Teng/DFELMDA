import numpy as np
from sklearn import preprocessing
from keras import utils

def prepare_data():
    print("loading data")
    SD = np.loadtxt("SD.txt", dtype=float, delimiter=" ")
    SM = np.loadtxt("SM.txt", dtype=float, delimiter=" ")
    A = np.loadtxt("interaction.txt", dtype=int, delimiter=" ")
    interacation = np.transpose(A)

    R_B = np.matlib.repmat(interacation, 495, 1)
    sm = np.repeat(SM, repeats=383, axis=0)
    train1 = np.concatenate((sm, R_B), axis=1)  # (189585,990)

    R_A = np.repeat(A, repeats=383, axis=0)
    sd = np.matlib.repmat(SD, 495, 1)
    train2 = np.concatenate((R_A, sd), axis=1)  # (189585,766)
    label = A.reshape((189585, 1))

    return train1, train2, label


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        f1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        f1_score = float(2 * tp) / ((2 * tp) + fp + fn)
    return acc, precision, sensitivity, specificity, MCC, f1_score


def transfer_array_format(data):  # data=X  , X= all the miRNA features, disease features
    formated_matrix1 = []
    formated_matrix2 = []
    for val in data:
        formated_matrix1.append(val[0])  # contains miRNA features ?
        formated_matrix2.append(val[1])  # contains disease features ?

    return np.array(formated_matrix1), np.array(formated_matrix2)


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = utils.np_utils.to_categorical(y)
    return y, encoder