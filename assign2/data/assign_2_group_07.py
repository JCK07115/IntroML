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
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score

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


def classify(model, K, r, train, alpha=0):
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


def validation(model, train, r):
    # train, validation, test = 

    reg = model.fit(train.iloc[:, :8], train['Concrete Compressive Strength'])
    R_square = r2_score(test['Concrete Compressive Strength'],
                        reg.predict(test.iloc[:, :8]))
    rse = RSE(reg, test.iloc[:, :8], test['Concrete Compressive Strength'])
    print("Validation:")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")


def CV(model, K, train):
    kf = KFold(n_splits=k, shuffle=True)
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
    split_ratios = [0.2, 0.3]
    for r in split_ratios:
        validation(train_df, r)

    # cross-validation
    K_set = [2, 3, 5, 10, 15, 20, 30, 50, 100]
    for K in K_set:
        classify(train_df, test_df, "Simple", K)


def Q2_results():
    ridge = Ridge()
    Alpha = [1, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05, 0.01]
    params = {'alpha': Alpha}
    X = train_df.iloc[:, :8]
    y = train_df['Concrete Compressive Strength']
    clf = GridSearchCV(ridge, params)
    clf.fit(X, y)
    alpha = clf.best_params_['alpha']
    reg = Ridge(alpha)
    reg.fit(X, y)
    X_test = test_df.iloc[:, :8]
    y_test = test_df['Concrete Compressive Strength']
    R_square = r2_score(y_test, reg.predict(X_test))
    rse = RSE(reg, X_test, y_test)
    print(f"Ridge Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")
    print("=========================================")
    Score = clf.cv_results_['mean_test_score']
    plt.figure()
    plt.plot(Alpha, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Ridge')
    plt.savefig(fname="Q2.png", format='png')


def Q3_results():
    lasso = Lasso()
    Alpha = [1, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01]
    params = {'alpha': Alpha}
    X = train_df.iloc[:, :8]
    y = train_df['Concrete Compressive Strength']
    clf = GridSearchCV(lasso, params)
    clf.fit(X, y)
    alpha = clf.best_params_['alpha']
    reg = Lasso(alpha)
    reg.fit(X, y)
    X_test = test_df.iloc[:, :8]
    y_test = test_df['Concrete Compressive Strength']
    R_square = r2_score(y_test, reg.predict(X_test))
    rse = RSE(reg, X_test, y_test)
    print(f"Lasso Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")
    print("=========================================")
    Score = clf.cv_results_['mean_test_score']
    plt.figure()
    plt.plot(Alpha, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Lasso')
    plt.savefig(fname="Q3.png", format='png')


def predictCompressiveStrength(Xtest, data_dir):
    ytest = None
    return ytest


if __name__ == "__main__":
    # Q1_results()
    Q2_results()
    Q3_results()
