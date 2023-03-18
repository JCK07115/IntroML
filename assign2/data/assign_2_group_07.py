"""
# Assignment 2
# Group 07
# ----------
# Chaoyue Xi                          : chaoyuex@mun.ca
# Mohammad Hamza Khan                 : mohammadhk@mun.ca
# Oluwafunmiwo Judah Sholola          : ojsholola@mun.ca
"""

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold

column_names = {
    "Cement_component1__kgInAM_3Mixture_": "Cement",
    "BlastFurnaceSlag_component2__kgInAM_3Mixture_": "Blast Furnace Slag",
    "FlyAsh_component3__kgInAM_3Mixture_": "Fly Ash",
    "Water_component4__kgInAM_3Mixture_": "Water",
    "Superplasticizer_component5__kgInAM_3Mixture_": "Superplasticizer",
    "CoarseAggregate_component6__kgInAM_3Mixture_": "Coarse Aggregate",
    "FineAggregate_component7__kgInAM_3Mixture_": "Fine Aggregate",
    "Age_day_": "Age",
    "ConcreteCompressiveStrength_MPa_Megapascals_":
    "Concrete Compressive Strength"
}

# creating the training and testing dataframes
train_df = pd.read_csv('./train.csv')
train_df.rename(columns=column_names, inplace=True)
test_df = pd.read_csv('./test.csv')
test_df.rename(columns=column_names, inplace=True)


def classify(train, test, model, k, alpha=0):
    reg = None
    if model == "Simple":
        reg = LinearRegression()
        print("Simple Linear Regression")
    if model == "Ridge":
        reg = Ridge(alpha=alpha)
        print(f"Ridge Regression(alpha={alpha})")
    if model == "Lasso":
        reg = Lasso(alpha=alpha)
        print(f"Lasso Regression(alpha={alpha})")
    if reg:
        validation(reg, train, test)
        CV(reg, train, k)
        print("=========================================")
    else:
        print("Unknown Error")


def RSE(model, pred, y_true):
    y_pred = model.predict(pred)
    RSS = np.sum(np.square(y_true - y_pred))
    return math.sqrt(RSS / (len(y_true) - 2))


def validation(model, train, test):
    reg = model.fit(train.iloc[:, :8], train['Concrete Compressive Strength'])
    R_square = reg.score(test.iloc[:, :8],
                         test['Concrete Compressive Strength'])
    rse = RSE(reg, test.iloc[:, :8], test['Concrete Compressive Strength'])
    print("Validation:")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")


def CV(model, train, k):
    kf = KFold(n_splits=k)
    X = train.iloc[:, :8]
    y = train['Concrete Compressive Strength']
    R_square = []
    rse = []
    for _, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train_folds = X.iloc[train_index]
        y_train_folds = y.iloc[train_index]
        X_test_folds = X.iloc[test_index]
        y_test_folds = y.iloc[test_index]
        reg = model.fit(X_train_folds, y_train_folds)
        R_square += [reg.score(X_test_folds, y_test_folds)]
        rse += [RSE(reg, X_test_folds, y_test_folds)]
    print(f"Cross Validation(n_splits={k}):")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")


def Q1_results():
    classify(train_df, test_df, "Simple", 3)


def Q2_results():
    classify(train_df, test_df, "Ridge", 3, alpha=0.5)


def Q3_results():
    classify(train_df, test_df, "Lasso", 3, alpha=0.5)


def predictCompressiveStrength(Xtest, data_dir):
    pass


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
