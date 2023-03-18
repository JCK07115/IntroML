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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

column_names = {
    "Cement_component1__kgInAM_3Mixture_": "Cement",
    "BlastFurnaceSlag_component2__kgInAM_3Mixture_": "BFSlag",
    "FlyAsh_component3__kgInAM_3Mixture_": "FlyAsh",
    "Water_component4__kgInAM_3Mixture_": "Water",
    "Superplasticizer_component5__kgInAM_3Mixture_": "SPlasticizer",
    "CoarseAggregate_component6__kgInAM_3Mixture_": "CoAggregate",
    "FineAggregate_component7__kgInAM_3Mixture_": "FiAggregate",
    "Age_day_": "Age",
    "ConcreteCompressiveStrength_MPa_Megapascals_": "CCStrength"
}

# creating the training and testing dataframes
train_df = pd.read_csv('./train.csv')
train_df.rename(columns=column_names, inplace=True)
test_df = pd.read_csv('./test.csv')
test_df.rename(columns=column_names, inplace=True)


def classify(model, K, tt_r, tv_r, train, alpha=0):
    reg = None
    if model == "Simple":
        reg = LinearRegression()
        print("Simple Linear Regression - [K={K}]")
    elif model == "Ridge":
        reg = Ridge(alpha=alpha)
        print(f"Ridge Regression - [K={K}, alpha={alpha}]")
    elif model == "Lasso":
        reg = Lasso(alpha=alpha)
        print(f"Lasso Regression - [K={K}, alpha={alpha}]")
    
    if reg:
        validation(reg, train, r)
        CV(reg, train, K)
        print("=========================================")
    else:
        print("Unknown Error")


def RSE(model, pred, y_true):
    y_pred = model.predict(pred)
    RSS = np.sum(np.square(y_true - y_pred))
    return math.sqrt(RSS / (len(y_true) - 2))


def validation(model, train, tt_r, tv_r):
    # train is now {tt_r} of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(train, train, test_size=1 - tt_r)

    # test is now {1-tt_r-(tv_r*0.5)} of the initial data set
    # validation is now {1-tt_r-(tv_r*0.5)} of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=tv_r) 

    print(len(x_train), len(x_val), len(x_test))

    model.fit(x_train.iloc[:, :8], x_train['CCStrength'])
    R_square = r2_score(test['CCStrength'],
                        model.predict(test.iloc[:, :8]))
    rse = RSE(model, test.iloc[:, :8], test['CCStrength'])
    print("Validation:")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")


def CV(model, K, train):
    kf = KFold(n_splits=K, shuffle=True)
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
        R_square += [r2_score(y_test_folds, reg.predict(X_test_folds))]
        rse += [RSE(reg, X_test_folds, y_test_folds)]

    print(f"Cross Validation(n_splits={K}):")
    print(f"RSE: {np.mean(rse)}")
    print(f"R^2: {np.mean(R_square)}")


def Q1_results():
    # validation (alpha's are ratio of )
    train_test_ratios = [0.2, 0.3, 0.5]
    test_val_ratios = [0.5]                 # setting validation and test to be exactly half of whatever remains from tt_r split (10%, 15%, 25% ea.)

    for tt_r in train_test_ratios:          # train_test_ratio
        for tv_r in test_val_ratios:        # test_val_ratio
            classify(train_df, tt_r, tv_r)

    # cross-validation
    K_set = [2, 3, 5, 10, 15, 20, 30, 50, 100]
    for K in K_set:
        classify("Simple", K, train_df)


def Q2_results():
    classify(train_df, test_df, "Ridge", 3, alpha=0.5)


def Q3_results():
    classify(train_df, test_df, "Lasso", 3, alpha=0.5)


def predictCompressiveStrength(Xtest, data_dir):
    pass


if __name__ == "__main__":
    Q1_results()
    # Q2_results()
    # Q3_results()
