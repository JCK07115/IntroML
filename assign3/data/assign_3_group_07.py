"""
# Assignment 3
# Group 07
# ----------
# Chaoyue Xi                          : chaoyuex@mun.ca
# Mohammad Hamza Khan                 : mohammadhk@mun.ca
# Oluwafunmiwo Judah Sholola          : ojsholola@mun.ca
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

columns = {
    0: "ctx-lh-inferiorparietal",
    1: "ctx-lh-inferiortemporal",
    2: "ctx-lh-isthmuscingulate",
    3: "ctx-lh-middletemporal",
    4: "ctx-lh-posteriorcingulate",
    5: "ctx-lh-precuneus",
    6: "ctx-rh-isthmuscingulate",
    7: "ctx-rh-posteriorcingulate",
    8: "ctx-rh-inferiorparietal",
    9: "ctx-rh-middletemporal",
    10: "ctx-rh-precuneus",
    11: "ctx-rh-inferiortemporal",
    12: "ctx-lh-entorhinal",
    13: "ctx-lh-supramarginal"
}


def get_data(data_dir):
    train_DAT = pd.read_csv(f'{data_dir}/train.fdg_pet.sDAT.csv', header=None)
    train_DAT.rename(columns=columns, inplace=True)
    train_DAT['label'] = 1
    train_NC = pd.read_csv(f'{data_dir}/train.fdg_pet.sNC.csv', header=None)
    train_NC.rename(columns=columns, inplace=True)
    train_NC['label'] = -1
    train_df = pd.concat([train_DAT, train_NC], axis=0)
    test_DAT = pd.read_csv(f'{data_dir}/test.fdg_pet.sDAT.csv', header=None)
    test_DAT.rename(columns=columns, inplace=True)
    test_DAT['label'] = 1
    test_NC = pd.read_csv(f'{data_dir}/test.fdg_pet.sNC.csv', header=None)
    test_NC.rename(columns=columns, inplace=True)
    test_NC['label'] = -1
    test_df = pd.concat([test_DAT, test_NC], axis=0)
    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label'])
    y_test = test_df['label']
    return X_train, y_train, X_test, y_test


def grid_search(params, X_train, y_train):
    clf = GridSearchCV(SVC(), params)
    clf.fit(X_train, y_train)
    return clf.best_params_, clf.cv_results_['mean_test_score']


def get_errors(svm, X_test, y_test):
    y_pred = svm.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ppv = tp / (tp + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    ba = (tpr + tnr) / 2
    print(f'accuracy: {acc}')
    print(f'sensitivity: {tpr}')
    print(f'specificity: {tnr}')
    print(f'precision: {ppv}')
    print(f'recall: {tpr}')
    print(f'balanced accuracy: {ba}')


def retrain_test(kernel,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 C,
                 d=3,
                 gamma='scale'):
    print(f"kernel: {kernel}")
    print(f"C: {C}")
    if kernel == 'linear':
        svm = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        svm = SVC(kernel=kernel, C=C, degree=d)
        print(f"d: {d}")
    elif kernel == 'rbf':
        svm = SVC(kernel=kernel, C=C, gamma=gamma)
        print(f"gamma: {gamma}")
    svm.fit(X_train, y_train)
    get_errors(svm, X_test, y_test)
    print("================================")


def classify(kernel, data_dir):
    X_train, y_train, X_test, y_test = get_data(data_dir)
    if kernel == 'linear':
        Cs = [0.1, 1, 10, 100, 1000]
        params = {'kernel': [kernel], 'C': Cs}
        best_params, Score = grid_search(params, X_train, y_train)
        C = best_params['C']
        retrain_test(kernel, X_train, y_train, X_test, y_test, C)
        plt.figure(figsize=(8, 6))
        plt.plot(Cs, Score)
        plt.xlabel('C')
        plt.ylabel('score')
        plt.title('Linear')
        plt.savefig(fname="Q1.png", format='png')
    elif kernel == 'poly':
        params = {
            'kernel': [kernel],
            'C': [0.1, 1, 10, 100, 1000],
            'degree': [2, 3, 4, 5]
        }
        best_params, _ = grid_search(params, X_train, y_train)
        C = best_params['C']
        d = best_params['degree']
        retrain_test(kernel, X_train, y_train, X_test, y_test, C, d=d)
    elif kernel == 'rbf':
        params = {
            'kernel': [kernel],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
        }
        best_params, _ = grid_search(params, X_train, y_train)
        C = best_params['C']
        gamma = best_params['gamma']
        retrain_test(kernel, X_train, y_train, X_test, y_test, C, gamma=gamma)
    else:
        print('Only support the following kernel: linear, poly, rbf')


def Q1_results():
    classify('linear', '.')


def Q2_results():
    classify('poly', '.')


def Q3_results():
    classify('rbf', '.')


def diagnoseDAT(Xtest, data_dir):
    trainDAT = pd.read_csv(f'{data_dir}/train.fdg_pet.sDAT.csv', header=None)
    trainDAT.rename(columns=columns, inplace=True)
    trainDAT['label'] = 1
    trainNC = pd.read_csv(f'{data_dir}/train.fdg_pet.sNC.csv', header=None)
    trainNC.rename(columns=columns, inplace=True)
    trainNC['label'] = -1
    trainDf = pd.concat([trainDAT, trainNC], axis=0)
    testDAT = pd.read_csv(f'{data_dir}/test.fdg_pet.sDAT.csv', header=None)
    testDAT.rename(columns=columns, inplace=True)
    testDAT['label'] = 1
    testNC = pd.read_csv(f'{data_dir}/test.fdg_pet.sNC.csv', header=None)
    testNC.rename(columns=columns, inplace=True)
    testNC['label'] = -1
    testDf = pd.concat([testDAT, testNC], axis=0)
    df = pd.concat([trainDf, testDf], axis=0)
    Xtrain = df.drop(columns=['label'])
    ytrain = df['label']
    svm = SVC(kernel='rbf', C=8.6, gamma=1.1)
    svm.fit(Xtrain, ytrain)
    return svm.predict(Xtest)


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
