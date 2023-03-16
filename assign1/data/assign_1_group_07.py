"""
# Assignment 1
# Group 07
# ----------
# Chaoyue Xi                          : chaoyuex@mun.ca
# Mohammad Hamza Khan                 : mohammadhk@mun.ca
# Oluwafunmiwo Judah Sholola          : ojsholola@mun.ca
"""

from numpy import arange, argmin, log1p, mean 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold, cross_val_score

from os import path
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import FunctionTransformer


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
        train_errs  -  
        test_errs   -
@desc: 
        generates a plot of the visualization boundary of the output feature labels
        of the training set's and test set's true labels, as well as the 
"""
def draw_plot(train_errs, test_errs, k_set, metric="Euclidean", rows=1, cols=1):
    fig, axs = plt.subplots(rows, cols, figsize=(30, 15))
    # plt.contour(X, Y)
    fig.suptitle(metric+" metric", fontsize=16)

    if rows>1 or cols>1:
        axs = axs.flatten()

        for i in range(len(k_set)):
            attr_id = 'pred_label_'+str(k_set[i])

            axs[i].scatter(grid_points_copy['x1'],
                    grid_points_copy['x2'],
                    c=grid_points_copy[attr_id].map(colors),
                    marker='.')
            axs[i].scatter(train_df['x1'],
                    train_df['x2'],
                    c=train_df['true_label'].map(colors),
                    marker='o')
            axs[i].scatter(test_df['x1'],
                    test_df['x2'],
                    c=test_df['true_label'].map(colors),
                    marker='+')
            axs[i].set(xlabel="$x_1$", ylabel="$x_2$", title = "k=" + str(k_set[i]) + ", Training Error: " +
                        str(train_errs[i]) + ", Test Error Rate: " + str(test_errs[i]))

    else:
        for i in range(len(k_set)):
            attr_id = 'pred_label_'+str(k_set[i])

            axs.scatter(grid_points_copy['x1'],
                    grid_points_copy['x2'],
                    c=grid_points_copy[attr_id].map(colors),
                    marker='.')
            axs.scatter(train_df['x1'],
                    train_df['x2'],
                    c=train_df['true_label'].map(colors),
                    marker='o')
            axs.scatter(test_df['x1'],
                    test_df['x2'],
                    c=test_df['true_label'].map(colors),
                    marker='+')
            axs.set(xlabel="$x_1$", ylabel="$x_2$", title = "k=" + str(k_set[i]) + ", Training Error: " +
                        str(train_errs[i]) + ", Test Error Rate: " + str(test_errs[i]))



"""
@params:
        k_set   - opt - list of k values to use as neighbours in KNN model, defaults to [30]
        metric  - opt - distance metric, defaults to Euclidean
@desc:
        performs classification using the KNNClassifiers model and
        the parameters specified as formal arguments into the function,
        or the defaults if no params are passed in

"""
def classify(train_df, test_df, k_set=[30], metric="Euclidean", save=False, filename='temp.png'):

    if metric == "Euclidean":
        p = 2
        m_name = 'Eucl'
    elif metric == "Manhattan":
        p = 1
        m_name = 'Manh'
    else:
        return "Invalid Metric"

    train_errs, test_errs = [], []
    
    global grid_points_copy
    grid_points_copy = grid_points.copy()

    for k in k_set:
        attr_id = 'pred_label_'+str(k)

        # create model with specific parameter, k, and fit
        model = KNeighborsClassifier(n_neighbors=k, p=p)
        model.fit(train_df[['x1', 'x2']], train_df['true_label'])

        # predict
        test_df['pred_labels_'+str(k)+'_'+m_name] = model.predict(test_df[['x1', 'x2']])
        grid_points_copy[attr_id] = model.predict(grid_points_copy[['x1', 'x2']])

        # calculate training and test error rates
        train_errs.append(
            round(
                1 - model.score(train_df[['x1', 'x2']], train_df['true_label']),
                4))
        test_errs.append(
            round(
                1 - model.score(test_df[['x1', 'x2']], test_df['true_label']),
                4))


    rows, cols = 1, 1
    if len(k_set)>1:
        rows = 2
        cols = int(len(k_set)/2 + len(k_set)%2)

    draw_plot(train_errs, test_errs, k_set, metric, rows, cols)

    if save:
       plt.savefig(filename, format='png')


    rows, cols = 1, 1
    if len(k_set)>1:
        rows = 2
        cols = int(len(k_set)/2 + len(k_set)%2)

    draw_plot(train_errs, test_errs, k_set, metric, rows, cols)

    if save:
       plt.savefig(filename, format='png')

    return train_errs, test_errs


def Q1_results(train_df, test_df):
    print('Generating results for Q1...')

    # predict using various k values
    train_errs_eucl, test_errs_eucl = classify(train_df, test_df, k_set=k_set, save=True, filename='Q1.png')
    print('mean_test_err_eucl:', mean(test_errs_eucl))


def Q2_results(train_df, test_df):
    print('Generating results for Q2...')

    # using the ideal value of k in k_set, re-predict, using the Manhattan distance metric
    train_errs_manh, test_errs_manh = classify(train_df, test_df, k_set=[30], metric="Manhattan", save=True, filename='Q2.png')
    print('nmean_test_err_manh:', mean(test_errs_manh))


def Q3_results(train_df, test_df):
    print('Generating results for Q3...')

    # using the metric that yields the lower (average) test error rate
    capacity = [1/k for k in k_set]
    train_errs, test_errs = classify(train_df, test_df, k_set=k_set, metric='Manhattan')

    _, err_plot = plt.subplots(figsize=(10, 10))
    err_plot.plot(capacity, train_errs, 'o-r', color='blue', label='Training')
    err_plot.plot(capacity, test_errs, 'o-r', color='red', label='Testing')
    err_plot.set_title('Manhattan'+", Error rate versus Model capacity")
    err_plot.set_xscale("log")
    err_plot.set(xlabel="1/k (log scale)", ylabel="Error rate")
    err_plot.legend()
    t1 = ("High Bias\n"
          "Low Variance\n"
          "Low Capacity\n"
          "Underfitting")
    plt.text(0.0050, 0.15, t1)
    t2 = ("Low Bias\n"
          "High Variance\n"
          "High Capacity\n"
          "Overfitting")
    plt.text(0.50, 0.15, t2)

    plt.savefig('Q3.png', format='png')


def improve_model(train_df):

    """plot raw data to gain insight into distributions"""
    def plot_histograms_boxplot_density(raw_data):
        cols = raw_data.columns
        fig, axs = plt.subplots(len(cols), 2, figsize=(30, 30))
        axs = axs.flatten()
        i=0

        for c in cols:
            print(type(raw_data[c][0])) 
            raw_data[c].plot.density(ax=axs[i], title=c)
            raw_data[c].plot.box(ax=axs[i+1])
            i+=2

    plot_histograms_boxplot_density(train_df.drop(columns=['true_label']))


    """we rid our training dataset of outliers in a bid to improve performance"""
    iqr_x1 = train_df['x2'].quantile(0.75) - train_df['x1'].quantile(0.25)
    lower_x1 = train_df['x2'].quantile(0.25) - (1.5 * iqr_x1)
    upper_x1 = train_df['x2'].quantile(0.75) + (1.5 * iqr_x1)
    train_df = train_df[(train_df['x1']>lower_x1) & (train_df['x1']<upper_x1)] 

    iqr_x2 = train_df['x2'].quantile(0.75) - train_df['x2'].quantile(0.25)
    lower_x2 = train_df['x2'].quantile(0.25) - (1.5 * iqr_x2)
    upper_x2 = train_df['x2'].quantile(0.75) + (1.5 * iqr_x2)
    train_df = train_df[(train_df['x2']>lower_x2) & (train_df['x2']<upper_x2)] 


    """re-plot to see if outliers are still present"""
    plot_histograms_boxplot_density(train_df.drop(columns=['true_label']))

    """scale data"""
    def scale_dataset(df, scalers):
        data = [0]*len(scalers)

        for i in range(len(scalers)):
            if scalers[i] == 'unscaled':
                data[i] = df.to_numpy()
                continue

            df_t = scalers[i].fit_transform(df.to_numpy())
            data[i] = df_t
    
        return data
    
    scalers = ['unscaled', MinMaxScaler(), RobustScaler(), FunctionTransformer(log1p)]
    scaled_data = scale_dataset(train_df.drop(columns=['true_label']), scalers)


    """plot scaled data to ensure that shape is preserved"""
    def plot_histograms_density_scalers(scaled_data, index, columns, scaler_labels, fig_title):
        fig, axs = plt.subplots(len(columns), len(scaler_labels),figsize=(30,30),constrained_layout=True)
        fig.suptitle(fig_title, fontsize=16)

        for s in range(len(scaler_labels)):
            df_scaled = pd.DataFrame(scaled_data[s], index=index, columns=columns)         # reconstructing dataframe from scaled data (np-array)

            for c in range(len(columns)):
                col = columns[c]
                df_scaled[col].hist(ax=axs[c, s], density=True)
                df_scaled[col].plot.density(ax=axs[c, s])
                axs[c, s].set_title(label=col+'_'+scaler_labels[s])
    
    scaler_labels = ['unscaled', 'MinMaxScaler()', 'RobustScaler()', 'FunctionTransformer(log1p)']
    plot_histograms_density_scalers(scaled_data, train_df.index, train_df.columns.drop('true_label'), scaler_labels, 'dist. of scaled data')
    

    """determine which scaling technique yields the lowest error rate on a KNN model based on results from Q1-Q3"""
    def run_model(scaled_data, scaler_labels, exp_output, splits, repeats, default_scorer):
        assert(len(scaled_data)==len(scaler_labels)), '0 > len(scaled_data)==len(scaler_labels)==True'

        scoring_dict = {'LinRegressor'              :   'r2',
                        'SGDRegressor'              :   'r2',
                        'SGDClassifier'             :   'f1_weighted',
                        'KNRegressor'               :   'r2',                                   
                        'KNClassifier'              :   'roc_auc',
                        'RFRegressor'               :   'r2',                           
                        'RFClassifier'              :   'accuracy',
                        'SVC'                       :   'f1_weighted',
                        'default-class'             :   'accuracy',                               
                        'default-reg'               :   'r2'                               
                        }
                        
        train_err_rates = [0]*len(scaler_labels)
        y = exp_output.to_numpy()
        s = splits                   # n_splits
        r = repeats                  # n_repeats
        cv = RepeatedKFold(n_splits=s, n_repeats=r, random_state=1)
        scorer = 'KNClassifier'
        model = KNeighborsClassifier(n_neighbors=30, metric='euclidean')

        for i in range(len(scaler_labels)):
            X_t = scaled_data[i]
            
            if default_scorer:
                scores = cross_val_score(model, X_t, y, scoring=scoring_dict['default-class'], cv=cv, n_jobs=-1)
            else:
                scores = cross_val_score(model, X_t, y, scoring=scoring_dict[scorer], cv=cv, n_jobs=-1)

            mean_err = round(1-mean(scores), 4)
            train_err_rates[i] = mean_err
            print('mean ' + scaler_labels[i] + ' training error rate: ', mean_err)

        return train_err_rates

    train_err_rates = run_model(scaled_data, scaler_labels, train_df['true_label'], 5, 3, False)
    min_train_err_ind = argmin(train_err_rates)
    best_scaled_data = pd.DataFrame(scaled_data[min_train_err_ind], index=train_df.index, columns=train_df.columns.drop('true_label'))


    """run grid search to find optimimum KNN parameters and further improve KNN model"""
    def grid_search(X_train, y_train, model, params):
        gs = GridSearchCV(model, params, cv=5, n_jobs=1, verbose=1, scoring='r2')
        gs.fit(X_train, y_train)
        print('Best Params:', gs.best_params_)

        return gs.best_params_
        
    metrics = ['euclidean', 'manhattan', 'minkowski', 'cosine', 
              'haversine', 'kulsinski', 'canberra', 'hamming', 
              'rogerstanimoto', 'p', 'russellrao', 'sokalmichener',
              'jaccard']
    grid_params = {
        'n_neighbors'   :   list(range(1,35)),
        'weights'       :   ['uniform', 'distance'],
        'metric'        :   metrics
    }
    hp_tuning = grid_search(best_scaled_data[['x1', 'x2']], train_df['true_label'], KNeighborsClassifier(), grid_params)
    
    model = KNeighborsClassifier(n_neighbors=hp_tuning['n_neighbors'], metric=hp_tuning['metric'], weights=hp_tuning['weights'])
    model.fit(best_scaled_data[['x1', 'x2']], train_df['true_label'])
    
    return model, hp_tuning


"""performing some experiments to improve the KNN model, and then
   retesting performance on the original testset
"""
def Q4_results(train_df, test_df, data_dir=path.abspath('.')):
    print('Generating results for Q4...')

    print('attempting to improve model...')
    revised_model, model_params = improve_model(train_df)

    print('using revised model on test.sDAT.csv and test.sNC.csv as "blind" test set... ')
    Xtest = test_df[['x1', 'x2']]
    ytest = diagnoseDAT(test_df[['x1', 'x2']], data_dir)
    print(ytest)

    Xtest_scaled = pd.DataFrame(FunctionTransformer(log1p).fit_transform(Xtest.to_numpy()),
                            index=Xtest.index,
                            columns=Xtest.columns)
    
    # realistically only the prof will have access to the true output labels for the actual 'blindset';
    # but since we are using the original test set as the 'blindset' we have access to its true output labels
    print('"Blind" test error: ', round(1 - revised_model.score(Xtest_scaled, test_df['true_label']), 4))


"""
@params:
        Xtest       -  N_test x 2 matrix of test feature vectors 
        data_dir    -  full path to the folder containing the following files:
                       train.sNC.csv, train.sDAT.csv, test.sNC.csv, test.sDAT.csv
@desc: 
        Returns a vector of predictions with elements "0" for sNC and "1" for
        sDAT, corresponding to each of the N_test features vectors in Xtest
        
"""
def diagnoseDAT(Xtest, data_dir=path.abspath('.')):

    # creating the training datasets using files in `data_dir`
    train_DAT = pd.read_csv(data_dir + '/train.sDAT.csv', header=None)
    train_DAT.rename(columns={0: "x1", 1: "x2"}, inplace=True)
    train_DAT['true_label'] = 1

    train_NC = pd.read_csv(data_dir + '/train.sNC.csv', header=None)
    train_NC.rename(columns={0: "x1", 1: "x2"}, inplace=True)
    train_NC['true_label'] = 0

    # merging the two training datasets
    train_df = pd.concat([train_DAT, train_NC], axis=0)

    # remove outliers
    iqr_x1 = train_df['x2'].quantile(0.75) - train_df['x1'].quantile(0.25)
    lower_x1 = train_df['x2'].quantile(0.25) - (1.5 * iqr_x1)
    upper_x1 = train_df['x2'].quantile(0.75) + (1.5 * iqr_x1)
    train_df = train_df[(train_df['x1']>lower_x1) & (train_df['x1']<upper_x1)] 

    iqr_x2 = train_df['x2'].quantile(0.75) - train_df['x2'].quantile(0.25)
    lower_x2 = train_df['x2'].quantile(0.25) - (1.5 * iqr_x2)
    upper_x2 = train_df['x2'].quantile(0.75) + (1.5 * iqr_x2)
    train_df = train_df[(train_df['x2']>lower_x2) & (train_df['x2']<upper_x2)] 

    # scaling using ideal scaler
    scaled_train_df = pd.DataFrame(FunctionTransformer(log1p).fit_transform((train_df.drop(columns=['true_label'])).to_numpy()), 
                                   index=train_df.index, 
                                   columns=train_df.columns.drop('true_label'))

    # we assign the hyperparameters of the model based on the improvement performed prior
    model = KNeighborsClassifier(metric='euclidean', n_neighbors=13, weights='uniform')
    model.fit(scaled_train_df[['x1', 'x2']], train_df['true_label'])
    
    # scale Xtest and run model on it to predict its labels
    Xtest_scaled = pd.DataFrame(FunctionTransformer(log1p).fit_transform(Xtest.to_numpy()),
                            index=Xtest.index,
                            columns=Xtest.columns)
    ytest = model.predict(Xtest_scaled)
    # print(ytest)
    # print(len(ytest))

    return ytest

if __name__ == "__main__":
    Q1_results(train_df, test_df)
    Q2_results(train_df, test_df)
    Q3_results(train_df, test_df)
    # Q4_results(train_df, test_df)
    # diagnoseDAT(test_df[['x1', 'x2']])
