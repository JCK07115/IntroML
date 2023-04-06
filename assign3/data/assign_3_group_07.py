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

# creating the training dataset
train_DAT = pd.read_csv('train.fdg_pet.sDAT.csv', header=None)
train_DAT.rename(columns=columns, inplace=True)
train_DAT['label'] = 1
train_NC = pd.read_csv('train.fdg_pet.sNC.csv', header=None)
train_NC.rename(columns=columns, inplace=True)
train_NC['label'] = -1
train_df = pd.concat([train_DAT, train_NC], axis=0)

# creating the test dataset
test_DAT = pd.read_csv('test.fdg_pet.sDAT.csv', header=None)
test_DAT.rename(columns=columns, inplace=True)
test_DAT['label'] = 1
test_NC = pd.read_csv('test.fdg_pet.sNC.csv', header=None)
test_NC.rename(columns=columns, inplace=True)
test_NC['label'] = -1
test_df = pd.concat([test_DAT, test_NC], axis=0)

X_train = train_df.drop(columns=['label'])
y_train = train_df['label']
X_test = test_df.drop(columns=['label'])
y_test = test_df['label']


def grid_search(params):
    clf = GridSearchCV(SVC(), params)
    clf.fit(X_train, y_train)
    return clf.best_params_, clf.cv_results_['mean_test_score']


def retrain_test(kernel, C, d=3, gamma='scale'):
    if kernel == 'linear':
        svm = SVC(kernel=kernel, C=C)
    elif kernel == 'poly':
        svm = SVC(kernel=kernel, C=C, degree=d)
    elif kernel == 'rbf':
        svm = SVC(kernel=kernel, C=C, gamma=gamma)
    svm.fit(X_test, y_test)


def classify(kernel):
    if kernel == 'linear':
        Cs = [0.1, 1, 10, 100, 1000]
        params = {'kernel': [kernel], 'C': Cs}
        best_params, Score = grid_search(params)
        C = best_params['C']
        retrain_test(kernel, C)
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
        best_params, _ = grid_search(params)
        C = best_params['C']
        d = best_params['degree']
        retrain_test(kernel, C, d=d)
    elif kernel == 'rbf':
        params = {
            'kernel': [kernel],
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
        }
        best_params, _ = grid_search(params)
        C = best_params['C']
        gamma = best_params['gamma']
        retrain_test(kernel, C, gamma=gamma)
    else:
        print('Only support the following kernel: linear, poly, rbf')


def Q1_results():
    classify('linear')


def Q2_results():
    classify('poly')


def Q3_results():
    classify('rbf')


def diagnoseDAT(Xtest, data_dir):
    ytest = None
    return ytest


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
