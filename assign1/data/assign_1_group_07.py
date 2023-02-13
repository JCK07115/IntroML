"""
# Assignment 1
# Group 07
# ----------
# Chaoyue Xi                          : chaoyuex@mun.ca
# Mohammad Hamza Khan                 : mohammadhk@mun.ca
# Oluwafunmiwo Judah Sholola          : ojsholola@mun.ca
"""

from numpy import argmin, mean
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# creating the training datasets
train_DAT = pd.read_csv('train.sDAT.csv', header=None)
train_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
train_DAT['true_label'] = 1

train_NC = pd.read_csv('train.sNC.csv', header=None)
train_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
train_NC['true_label'] = 0

# merging the two training datasets
train_df = pd.concat([train_DAT, train_NC], axis=0)

# creating the test datasets
test_DAT = pd.read_csv('test.sDAT.csv', header=None)
test_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
test_DAT['true_label'] = 1

test_NC = pd.read_csv('test.sNC.csv', header=None)
test_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
test_NC['true_label'] = 0

# merging the two testing datasets
test_df = pd.concat([test_DAT, test_NC], axis=0)

grid_points = pd.read_csv('2D_grid_points.csv', header=None)
grid_points.rename(columns={0: "x1", 1: "x2"}, inplace=True)

k_set = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
colors = {0: 'green', 1: "blue"}
"""
@params:
        k_set   - opt - list of k values to use as neighbours in KNN model,
                        defaults to [5]
        metric  - opt - distance metric, defaults to Euclidean
@desc:
        performs classification using the KNNClassifiers model and
        the parameters specified as formal arguments into the function,
        or the defaults if no params are passed in

"""


def classify(k_set=[5], metric="Euclidean"):
    name = ''
    if metric == "Euclidean":
        p = 2
        name = 'Eucl'
    elif metric == "Manhattan":
        p = 1
        name = 'Manh'
    else:
        return "Invalid Metric"

    train_errs, test_errs = [], []

    for k in k_set:
        # create model with specific parameter, k, and fit
        model = KNeighborsClassifier(n_neighbors=k, p=p)
        model.fit(train_df[['x1', 'x2']], train_df['true_label'])

        # predict
        test_df['pred_labels_' + str(k) + '_' + name] = model.predict(
            test_df[['x1', 'x2']])
        grid_points['pred_label'] = model.predict(grid_points[['x1', 'x2']])

        # calculate training and test error rates
        train_errs.append(
            round(
                1 -
                model.score(train_df[['x1', 'x2']], train_df['true_label']),
                4))
        test_errs.append(
            round(
                1 - model.score(test_df[['x1', 'x2']], test_df['true_label']),
                4))

    return train_errs, test_errs


"""
@params:
        train_errs  -
        test_errs   -
@desc:
        generates a plot of the visualization boundary of the output feature
        labels of the training set's and test set's true labels
"""


def draw_plot(train_errs,
              test_errs,
              k_set,
              metric="Euclidean",
              rows=2,
              cols=1):
    fig, axs = plt.subplots(rows, cols, figsize=(30, 15))
    fig.suptitle(metric + " metric", fontsize=16)
    axs = axs.flatten()

    for i in range(len(k_set)):
        axs[i].scatter(grid_points['x1'],
                       grid_points['x2'],
                       c=grid_points['pred_label'].map(colors),
                       marker='.')
        axs[i].scatter(train_df['x1'],
                       train_df['x2'],
                       c=train_df['true_label'].map(colors),
                       marker='o')
        axs[i].scatter(test_df['x1'],
                       test_df['x2'],
                       c=test_df['true_label'].map(colors),
                       marker='+')
        axs[i].set(xlabel="$x_1$",
                   ylabel="$x_2$",
                   title="k=" + str(k_set[i]) + ", Training Error: " +
                   str(train_errs[i]) + ", Testing Error Rate: " +
                   str(test_errs[i]))


def Q1_results():
    print('Generating results for Q1...')

    # predict using various k values
    train_errs_eucl, test_errs_eucl = classify(k_set)
    global mean_eucl_test_err
    mean_eucl_test_err = mean(test_errs_eucl)

    # store index of min test error using Eucl
    global ind_min_test_err_eucl
    ind_min_test_err_eucl = argmin(test_errs_eucl)

    # plot results and save
    draw_plot(train_errs_eucl, test_errs_eucl, k_set, rows=2, cols=5)
    plt.savefig('Q1.png', format='png')
    # plt.show()


def Q2_results():
    print('Generating results for Q2...')

    # using the ideal value of k in k_set, re-predict, and store
    # the mean of the test errors using the Manhattan distance metric
    train_errs_manh, test_errs_manh = classify([k_set[ind_min_test_err_eucl]],
                                               "Manhattan")
    global mean_manh_test_err
    mean_manh_test_err = mean(test_errs_manh)

    # plot results and save
    draw_plot(train_errs_manh,
              test_errs_manh, [k_set[ind_min_test_err_eucl]],
              metric="Manhattan")
    plt.savefig('Q2.png', format='png')
    # plt.show()


def Q3_results():
    print('Generating results for Q3...')

    # determining which metric yields the lower (average) test error rate
    metric = 'Euclidean'
    if mean_manh_test_err < mean_eucl_test_err:
        metric = 'Manhattan'

    capacity = [1 / k for k in k_set]
    train_errs, test_errs = classify(k_set, metric)

    _, err_plot = plt.subplots(figsize=(10, 10))
    err_plot.plot(capacity, train_errs, color='blue', label='Training')
    err_plot.plot(capacity, test_errs, color='red', label='Testing')
    err_plot.set_xscale("log")
    err_plot.legend()
    err_plot.set_title(metric + ", Error rate versus Model capacity")
    plt.savefig('Q3.png', format='png')
    # plt.show()


def grid_search(X_train, y_train, model, params):
    gs = GridSearchCV(model, params, cv=5, n_jobs=1, verbose=1, scoring='r2')
    gs.fit(X_train, y_train)
    print('Best Params:', gs.best_params_)

    return gs.best_params_


def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1" for
    sDAT, corresponding to each of the N_test features vectors in Xtest
    Xtest N_test x 2 matrix of test feature vectors
    data_dir full path to the folder containing the following files:
    train.sNC.csv, train.sDAT.csv, test.sNC.csv, test.sDAT.csv
    """

    # creating the training datasets using files in `data_dir`
    train_DAT = pd.read_csv(data_dir + '/train.sDAT.csv', header=None)
    train_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
    train_DAT['true_label'] = 1

    train_NC = pd.read_csv(data_dir + '/train.sNC.csv', header=None)
    train_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
    train_NC['true_label'] = 0

    # merging the two training datasets
    train_df = pd.concat([train_DAT, train_NC], axis=0)
    # print(train_df['true_label'])

    grid_params = {
        'n_neighbors': list(range(10, 35)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    hp_tuning = grid_search(train_df[['x1', 'x2']], train_df['true_label'],
                            KNeighborsClassifier(), grid_params)
    model = KNeighborsClassifier(metric=hp_tuning['metric'],
                                 n_neighbors=hp_tuning['n_neighbors'],
                                 weights=hp_tuning['weights'])
    model.fit(train_df[['x1', 'x2']], train_df['true_label'])
    ytest = model.predict(Xtest)

    print(
        'train_error: ',
        round(1 - model.score(train_df[['x1', 'x2']], train_df['true_label']),
              4))
    print('test_error: ',
          round(1 - model.score(Xtest, test_df['true_label']), 4))

    return ytest


def Q4_results(data_dir=path.abspath('.')):
    print('Generating results for Q4...')

    Xtest = test_df[['x1', 'x2']]
    ytest = diagnoseDAT(Xtest, data_dir)

    print(ytest)
    print(type(ytest))


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
    Q4_results()
