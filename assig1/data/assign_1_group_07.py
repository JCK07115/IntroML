#
# Assignment 1
#
# Group 07
# Chaoyue Xi chaoyuex@mun.ca
# Oluwafunmiwo Judah Sholola
#
#  Mohammad Hamza Khan mohammadhk@mun.ca

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# creating the training datasets
train_DAT = pd.read_csv('train.sDAT.csv', header=None)
train_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
train_DAT['label'] = 1

train_NC = pd.read_csv('train.sNC.csv', header=None)
train_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
train_NC['label'] = 0

# merging the two training datasets
train_df = pd.concat([train_DAT, train_NC], axis=0)

# creating the test datasets
test_DAT = pd.read_csv('test.sDAT.csv', header=None)
test_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
test_DAT['label'] = 1

test_NC = pd.read_csv('test.sNC.csv', header=None)
test_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
test_NC['label'] = 0

# merging the two testing datasets
test_df = pd.concat([test_DAT, test_NC], axis=0)

grid_points = pd.read_csv('2D_grid_points.csv', header=None)
grid_points.rename(columns={0: "x1", 1: "x2"}, inplace=True)

k = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
colors = {0: 'green', 1: "blue"}
capacity = []
for i in k:
    capacity.append(1 / i)


def classify(k=30, metric="Euclidean", plot=True):
    if metric == "Euclidean":
        p = 2
    elif metric == "Manhattan":
        p = 1
    else:
        return "Invalid Metric"
    neigh = KNeighborsClassifier(n_neighbors=k, p=p)
    neigh.fit(train_df[['x1', 'x2']], train_df['label'])
    if plot:
        draw_plot(neigh, k, metric)
        plt.savefig('classify.pdf', format='pdf')
        plt.show()
    return neigh


def draw_plot(neigh, k, metric="Euclidean"):
    train_err = round(
        1 - neigh.score(train_df[['x1', 'x2']], train_df['label']), 3)
    test_err = round(1 - neigh.score(test_df[['x1', 'x2']], test_df['label']),
                     3)
    grid_points_copy = grid_points.copy()
    grid_points_copy['predict'] = neigh.predict(grid_points)
    _, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(grid_points_copy['x1'],
               grid_points_copy['x2'],
               c=grid_points_copy['predict'].map(colors),
               marker='.')
    ax.scatter(train_df['x1'],
               train_df['x2'],
               c=train_df['label'].map(colors),
               marker='o')
    ax.scatter(test_df['x1'],
               test_df['x2'],
               c=test_df['label'].map(colors),
               marker='+')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(metric + ", k=" + str(k) + ", Training Error Rate: " +
                 str(train_err) + ", Testing Error Rate: " + str(test_err))


def Q1_results():
    for i in k:
        neigh = classify(k=i, plot=False)
        draw_plot(neigh, i)
        plt.savefig('Q1_' + str(i) + '.pdf', format='pdf')
        plt.show()


def Q2_results():
    neigh = classify(k=30, metric="Manhattan", plot=False)
    draw_plot(neigh, 30, "Manhattan")
    plt.savefig('Q2.pdf', format='pdf')
    plt.show()


def Q3_results():
    train_list = []
    test_list = []
    for i in k:
        neigh = classify(k=i, plot=False)
        train_err = round(
            1 - neigh.score(train_df[['x1', 'x2']], train_df['label']), 3)
        test_err = round(
            1 - neigh.score(test_df[['x1', 'x2']], test_df['label']), 3)
        train_list.append(train_err)
        test_list.append(test_err)
    _, err_plot = plt.subplots(figsize=(10, 10))
    err_plot.plot(capacity, train_list, 'o-r', color='blue', label='Training')
    err_plot.plot(capacity, test_list, 'o-r', color='red', label='Testing')
    err_plot.set_xscale("log")
    err_plot.legend()
    err_plot.set_title("Euclidean, Error rate versus Model capacity")
    plt.savefig('Q3.pdf', format='pdf')
    plt.show()

def grid_search(X_train, y_train, model, params):
    gs = GridSearchCV(model, params, cv=5, n_jobs=1, verbose=1, scoring='r2')
    gs.fit(X_train, y_train)
    print('Best Params:', gs.best_params_)
    return gs.best_params_

hp_tuning_res = grid_search(df_fs_scaled_price, target_price_y, KNeighborsRegressor(), grid_params)

def diagnoseDAT(Xtest, data_dir):
    """Returns a vector of predictions with elements "0" for sNC and "1" for
    sDAT, corresponding to each of the N_test features vectors in Xtest
    Xtest N_test x 2 matrix of test feature vectors
    data_dir full path to the folder containing the following files:
    train.sNC.csv, train.sDAT.csv, test.sNC.csv, test.sDAT.csv
    """
    pass  # TODO: Hello Judah. The doc above is from the assignment guide. Please follow it and do your work here. XD

    grid_params = {
        'n_neighbors'   :   list(range(1,30)),
        'weights'       :   ['uniform', 'distance'],
        'metric'        :   ['euclidean', 'manhattan', 'minkowski']
    }



if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()

