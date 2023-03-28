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
train_df = pd.read_csv('train.csv')
train_df.rename(columns=column_names, inplace=True)
test_df = pd.read_csv('test.csv')
test_df.rename(columns=column_names, inplace=True)

X_train = train_df.drop(columns=['CCStrength'])
y_train = train_df['CCStrength']
X_test = test_df.drop(columns=['CCStrength'])
y_test = test_df['CCStrength']


def classify(type, tt_rs=None, tv_rs=None, K_set=None, alpha=0):
    model = None
    if type == "Simple":
        model = LinearRegression()
        if K_set==None:
            print("Simple Linear Regression")
    elif type == "Ridge":
        model = Ridge(alpha=alpha)
        # print(f"Ridge Regression : [K={K}, alpha={alpha}]")
    elif type == "Lasso":
        model = Lasso(alpha=alpha)
        # print(f"Lasso Regression : [K={K}, alpha={alpha}]")

    if model and tt_rs!=None:       # validation approach
        val_models = []
        val_models_rse = []
        val_models_rsq = []
        val_models_config = []
        
        # model selection
        for tt_r in tt_rs:  # train_test_ratio
            for tv_r in tv_rs:  # test_val_ratio
                model, score = validation(model, tt_r, tv_r)
                val_models.append(model)
                val_models_rse.append(score[0])
                val_models_rsq.append(score[1])
                val_models_config.append([tt_r, tv_r])
                print("-----------------------------------------")

        # model assessment
        opt_per_ind = np.argmin(val_models_rse)              # could also use argmax(val_models_rsq) to find optimal model
        opt_model = val_models[opt_per_ind]
        opt_tt_r = val_models_config[opt_per_ind][0]
        opt_tv_r = val_models_config[opt_per_ind][1]
        validation(opt_model, opt_tt_r, opt_tv_r, test=True)

        print("=========================================")
    elif model and K_set!=None:     # cross-validation approach
        opt_model = None
        min_RSE = 10000000

        # model selection
        for K in K_set:
            print(f"{type} Linear Regression : [K={K}]")

            model, rse = CV(model, K)
            if rse < min_RSE:
                min_RSE = rse
                opt_model = model

            print("=========================================")
        
        # model assessment
        opt_model.fit(X_train, y_train)
        rse = RSE(opt_model, X_test, y_test)
        rsquare = RSq(opt_model, X_test, y_test)
        print(f"RSE_cv_test: {rse}")
        print(f"RSquare_cv_test: {rsquare}")

    else:
        print("Unknown Error")


def RSq(model, x_test, y_true):
    return r2_score(y_true, model.predict(x_test))


def RSE(model, x_test, y_true):
    y_pred = model.predict(x_test)
    RSS = np.sum(np.square(y_true - y_pred))
    return math.sqrt(RSS / (len(y_true) - 2))


def validation(model, tt_r, tv_r, test=False):
    # train is now {tt_r} of the entire data set
    x_train_sub, x_test_sub, y_train_sub, y_test_sub = train_test_split(X_train,
                                                        y_train,
                                                        test_size=1 - tt_r)
    
    # test is now {1-tt_r-(tv_r*0.5)} of the initial data set
    # validation is now {1-tt_r-(tv_r*0.5)} of the initial data set
    x_val_sub, x_test_sub, y_val_sub, y_test_sub = train_test_split(x_test_sub,
                                                    y_test_sub,
                                                    test_size=tv_r)
    
    if not test:
        print(f"x_train_sub: {len(x_train_sub)}, y_train_sub: {len(y_train_sub)}")
        print(f"x_val_sub: {len(x_val_sub)}, y_val_sub: {len(y_val_sub)}")
        # print(f"x_test_sub: {len(x_test_sub)}, y_test_sub: {len(y_test_sub)}\n")

        model.fit(x_train_sub, y_train_sub)
        rse = RSE(model, x_val_sub, y_val_sub)
        rsquare = RSq(model, x_val_sub, y_val_sub)
    
        print("Validation:")
        print(f"RSE_val_sub: {rse}")
        print(f"RSquare_val_sub: {rsquare}")
        return model, [rse, rsquare]
    else:
        x_train_sub = pd.concat([x_train_sub, x_val_sub], axis=0)
        y_train_sub = pd.concat([y_train_sub, y_val_sub], axis=0)

        print(f"x_train_sub: {len(x_train_sub)}, y_train_sub: {len(y_train_sub)}")
        print(f"x_test_sub: {len(x_test_sub)}, y_test_sub: {len(y_test_sub)}\n")

        model.fit(x_train_sub, y_train_sub)
        rse = RSE(model, x_test_sub, y_test_sub)
        rsquare = RSq(model, x_test_sub, y_test_sub)
    
        print("Test:")
        print(f"RSE_test_sub: {rse}")
        print(f"RSquare_test_sub: {rsquare}")
        return model, [rse, rsquare]

def CV(model, K):
    kf = KFold(n_splits=K, shuffle=True)
    RSq_vals = []
    RSE_vals = []
    min_RSE = 1000000000
    opt_model = None

    # model selection
    for _, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
        X_train_folds = X_train.iloc[train_index]
        y_train_folds = y_train.iloc[train_index]
        X_test_folds = X_train.iloc[test_index]
        y_test_folds = y_train.iloc[test_index]
        model.fit(X_train_folds, y_train_folds)
        rse = RSE(model, X_test_folds, y_test_folds)
        rsquare = RSq(model, X_test_folds, y_test_folds)
        RSE_vals.append(rse)
        RSq_vals.append(rsquare)

        if rse < min_RSE:
            min_RSE = rse
            opt_model = model

    print(f"Cross Validation(n_splits={K}):")
    print(f"RSE_cv_fold: {np.mean(RSE_vals)}")
    print(f"RSquare_cv_fold: {np.mean(RSq_vals)}")

    return opt_model, min_RSE 



def Q1_results():
    # validation (alpha's are ratio of training subset to test subset)
    train_test_ratios = [0.2, 0.3, 0.5]
    test_val_ratios = [0.5]  # setting validation and test subsets to be exactly half of whatever remains from tt_r split (10%, 15%, 25% ea.)

    classify(type="Simple",
             tt_rs=train_test_ratios,
             tv_rs=test_val_ratios)

    # cross-validation
    K_set = [2, 3, 5, 10, 15, 20, 30, 50, 100]
    classify(type="Simple", 
             K_set=K_set)


def Q2_results():
    # Set GridSearchCV
    ridge = Ridge()
    alphas = [1, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.7, 0.5, 0.2, 0.01]
    params = {'alpha': alphas}
    clf = GridSearchCV(ridge, params)
    clf.fit(X_train, y_train)

    # Train and test model with best alpha
    alpha = clf.best_params_['alpha']
    model = Ridge(alpha)
    model.fit(X_train, y_train)
    rse = RSE(model, X_test, y_test)
    rsquare = RSq(model, X_test, y_test)
    print(f"Ridge Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"RSquare: {rsquare}")
    print("=========================================")

    # Generate plot
    Score = clf.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Ridge')
    plt.savefig(fname="Q2.png", format='png')


def Q3_results():
    # Set GridSearchCV
    lasso = Lasso()
    alphas = [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.0875, 0.075, 0.05, 0.01]
    params = {'alpha': alphas}
    clf = GridSearchCV(lasso, params)
    clf.fit(X_train, y_train)

    # Train and test model with best alpha
    alpha = clf.best_params_['alpha']
    model = Lasso(alpha)
    model.fit(X_train, y_train)
    rse = RSE(model, X_test, y_test)
    rsquare = RSq(model, X_test, y_test)
    print(f"Lasso Regression - alpha={alpha}")
    print(f"RSE: {rse}")
    print(f"RSquare: {rsquare}")
    print("=========================================")

    # Generate plot
    Score = clf.cv_results_['mean_test_score']
    plt.figure(figsize=(8, 6))
    plt.plot(alphas, Score)
    plt.xlabel('alpha')
    plt.ylabel('score')
    plt.title('Lasso')
    plt.savefig(fname="Q3.png", format='png')


def predictCompressiveStrength(Xtest, data_dir):
    # Prepare the training dataset
    trainDf = pd.read_csv(f'{data_dir}/train.csv')
    trainDf.rename(columns=column_names, inplace=True)
    testDf = pd.read_csv(f'{data_dir}/test.csv')
    testDf.rename(columns=column_names, inplace=True)
    df = pd.concat([trainDf, testDf], axis=0)
    Xtrain = df.drop(columns=['CCStrength'])
    ytrain = df['CCStrength']

    # Train the model.
    # Model: Polynomial regression with degree=3
    poly = PolynomialFeatures(degree=3, include_bias=False)
    train_poly = poly.fit_transform(Xtrain, ytrain)
    model = LinearRegression()
    model.fit(train_poly, ytrain)

    # Predict the test value
    test_poly = poly.fit_transform(Xtest)
    return model.predict(test_poly)

if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    # predictCompressiveStrength()
