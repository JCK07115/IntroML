"""
# Assignment 3
# Group 07
# ----------
# Chaoyue Xi                          : chaoyuex@mun.ca
# Mohammad Hamza Khan                 : mohammadhk@mun.ca
# Oluwafunmiwo Judah Sholola          : ojsholola@mun.ca
"""

import pandas as pd

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


def classify():
    pass


def Q1_results():
    pass


def Q2_results():
    pass


def Q3_results():
    pass


def diagnoseDAT(Xtest, data_dir):
    ytest = None
    return ytest


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
