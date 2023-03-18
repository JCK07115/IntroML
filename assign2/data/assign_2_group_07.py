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
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

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

X_train = train_df.drop(columns=['CCStrength'])
y_train = train_df['CCStrength']
X_test = test_df.drop(columns=['CCStrength'])
y_test = test_df['CCStrength']


def classify(model, K, tt_rs, tv_rs, train, alpha=0):
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
        val_models = []
        val_models_scores = []
        for tt_r in tt_rs:          # train_test_ratio
            for tv_r in tv_rs:        # test_val_ratio
                mod, score = validation(model, train, tt_r, tv_r)



        CV(reg, train, K)
        print("=========================================")
    else:
        print("Unknown Error")


def RSq(model, x_test, y_true):
    return r2_score(y_true, model.predict(x_test))

def RSE(model, x_test, y_true):
    y_pred = model.predict(x_test)
    RSS = np.sum(np.square(y_true - y_pred))
    return math.sqrt(RSS / (len(y_true) - 2))


def validation(model, train, tt_r, tv_r):
    # train is now {tt_r} of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1 - tt_r)

    # test is now {1-tt_r-(tv_r*0.5)} of the initial data set
    # validation is now {1-tt_r-(tv_r*0.5)} of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=tv_r) 

    # print(len(x_train), len(x_val), len(x_test))

    model.fit(x_train, y_train)
    rse = RSE(model, x_val, y_val)
    rsquare = RSq(model, x_val, y_val)
    print("Validation:")
    print(f"RSE: {rse}")
    print(f"R^2: {rsquare}")
    return model


def CV(model, K, train):
    kf = KFold(n_splits=K, shuffle=True)
    # X = train.iloc[:, :8]
    # y = train['Concrete Compressive Strength']
    RSq_vals = []
    RSE_vals = []

    for _, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        X_test_folds = X_train.iloc[test_index]
        y_test_folds = y_train.iloc[test_index]
        # reg = model.fit(X_train_folds, y_train_folds)
        model.fit(X_train_folds, y_train_folds)
        RSq_vals += [RSq(model, X_test_folds, y_test_folds)]
        RSE_vals += [RSE(model, X_test_folds, y_test_folds)]

    print(f"Cross Validation(n_splits={K}):")
    print(f"RSE: {np.mean(RSq_vals)}")
    print(f"R^2: {np.mean(RSE_vals)}")


def Q1_results():
    # validation (alpha's are ratio of )
    train_test_ratios = [0.2, 0.3, 0.5]
    test_val_ratios = [0.5]                 # setting validation and test to be exactly half of whatever remains from tt_r split (10%, 15%, 25% ea.)

    classify(model="Simple", tt_rs=train_test_ratios, tv_rs=test_val_ratios, train=train_df)

    # cross-validation
    K_set = [2, 3, 5, 10, 15, 20, 30, 50, 100]
    for K in K_set:
        classify(model="Simple", K=K, train=train_df)


def Q2_results():
    # Set GridSearchCV
    ridge = Ridge()
    Alpha = [1, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.7, 0.5, 0.2, 0.01]
    params = {'alpha': Alpha}
    clf = GridSearchCV(ridge, params)
    clf.fit(X_train, y_train)

    # Train and test model with best alpha
    alpha = clf.best_params_['alpha']
    reg = Ridge(alpha)
    reg.fit(X_train, y_train)
    R_square = r2_score(y_test, reg.predict(X_test))
    rse = RSE(reg, X_test, y_test)
    print(f"Ridge Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")
    print("=========================================")

    # Generate plot
    Score = clf.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 6))
    plt.plot(Alpha, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Ridge')
    plt.savefig(fname="Q2.png", format='png')


def Q3_results():
    # Set GridSearchCV
    lasso = Lasso()
    Alpha = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.0875, 0.075, 0.05, 0.01]
    params = {'alpha': Alpha}
    clf = GridSearchCV(lasso, params)
    clf.fit(X_train, y_train)

    # Train and test model with best alpha
    alpha = clf.best_params_['alpha']
    reg = Lasso(alpha)
    reg.fit(X_train, y_train)
    R_square = r2_score(y_test, reg.predict(X_test))
    rse = RSE(reg, X_test, y_test)
    print(f"Lasso Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"R^2: {R_square}")
    print("=========================================")

    # Generate plot
    Score = clf.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 6))
    plt.plot(Alpha, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Lasso')
    plt.savefig(fname="Q3.png", format='png')


def predictCompressiveStrength(Xtest=None, data_dir=None):
    train_poly = PolynomialFeatures(3).fit_transform(X_train, y_train)
    test_poly = PolynomialFeatures(3).fit_transform(X_test, y_test)
    # ytest = None
    # return ytest


if __name__ == "__main__":
    # Q1_results()
    Q2_results()
    Q3_results()
