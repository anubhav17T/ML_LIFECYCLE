# """BUILDING MACHINE LEARNING PIPELINES: DATA ANALYTICS AND ENGINEERING"""
# """LIFECYCLE :-
# DATA ANALYSIS(EDA)
# FEATURE ENGINEERING
# FEATURE SELECTION
# MODEL BUILDNG
# MODEL DEPLOYMENT
# TESTING AND MONITORING MODEL DEPLOYMENT
# """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("/Users/anubhavtyagi/Desktop/train.csv", header="infer")
""" BASIC EDA """
print(dataset.head(n=10))
print(dataset.tail(n=10))
print(dataset.shape)
print(dataset.isnull().sum())
print(dataset.describe(include="object"))
print(dataset.describe())

print(dataset.dtypes)

"""FIND MISSING VALUES AND PERCENTAGE OF MISSING VALUES AND COLUMNS HAVING MISSING VALUES"""
feature = []
for columns in dataset.columns:
    if dataset[columns].isnull().sum() >= 1:
        col = {"column": columns, "missing_count": dataset[columns].isnull().sum(),
               "percentage": np.round(dataset[columns].isnull().mean(), 2)}
        feature.append(col)
print(feature)

"""NOW OUR DATASET HAS MISSING VALUES THEREFORE WE WILL NEED TO FIND WETHER THE MISSING VALUE HAS IMAPACT ON THE 
TARGET COLUMNS OR NOT (BY THIS WE CAN FIND IS THERE ANY RELATION BETWEEN MISSING VALUES AND THE DEPENDENT VARIABLES,
So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section """
# ALSO WE ARE PLOTTING MEDIAN BECAUSE WE ARE ASSUMING THAT WE HAVE OUTLIER IN THE DATASET

# feature1 = [features for features in dataset.columns if dataset[features].isnull().sum()>=1]
# data = dataset.copy()
# for cols in feature1:
# let's make a variable that indicates 1 if the observation was missing or zero otherwise
#     data[cols] = np.where(data[cols].isnull(), 1, 0)
# let's calculate the mean SalePrice where the information is missing or present
#     data.groupby(cols)["SalePrice"].median().plot.bar()
#     plt.title(feature)
#     plt.show()
#


""" ANALYSING NUMERIC VARIABLES """
numerical_feature = [features for features in dataset.columns if dataset[features].dtype != 'O']
# total numerical features
print(numerical_feature)
# number of numerical features
print(len(numerical_feature))

""" IN NUMERICAL FEATURE WE HAVE ORIDINAL,DISCRETTE AND CONTINOUS VARIABLE"""
# FINDING DISCREETE FEATURE

discreete_feature = []
for discreete in numerical_feature:
    if len(dataset[discreete].unique()) <= 25:
        discreete_feature.append(discreete)
print(discreete_feature)

continous_var = []
for cont in numerical_feature:
    if cont not in discreete_feature:
        continous_var.append(cont)
print(len(continous_var))
print(continous_var)

# NOW SAMEWAY WE CAN FIND THE RELATIONSHIP BETWEEN THE TARGET VAR AND DISCRETE VARIABLES AND WE LOOK OUT FOR EXPONENTIAL RELATIONSHIP
# for feature in discreete_feature:
#     data = dataset.copy()
#     data.groupby(feature)['SalePrice'].median().plot.bar()
#     plt.xlabel(feature)
#     plt.ylabel('SalePrice')
#     plt.title(feature)
#     plt.show()

""" ANALYSE CATEGORICAL FEATURE AND FEATURE CATEGORIES"""
categorical_feature = [category for category in dataset.columns if dataset[category].dtype =="O"]
for i in categorical_feature:
    dataset[i].value_counts(normalize=True)