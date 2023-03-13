import pandas as pd
from sklearn.linear_model import LinearRegression

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


def classify(train, test, model="Simple", alpha=0):
    if model == "Simple":
        reg = LinearRegression().fit(train.iloc[:, :8],
                                     train['Concrete Compressive Strength'])
        return reg.score(test.iloc[:, :8],
                         test['Concrete Compressive Strength'])
    if model == "Ridge":
        return "TODO"
    if model == "Lasso":
        return "TODO"


def Q1_results():
    pass


def Q2_results():
    pass


def Q3_results():
    pass


def predictCompressiveStrength(Xtest, data_dir):
    pass


if __name__ == "__main__":
    Q1_results()
    Q2_results()
    Q3_results()
