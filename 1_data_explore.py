import copy
import os
import numpy as np

import math
import pandas
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy.random as random
from pandas import DataFrame
from datetime import datetime
import numpy
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#create parses for formatted date data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')

#Load the dataset from CSV
try:
    data_all = pd.read_csv("dataset_mood_smartphone.csv", index_col=0, parse_dates=['time'], header=0, date_parser=dateparse)
except IOError as e:
    print('File not found!')
    raise e

# get a list of unique variable names
variables = data_all['variable'].unique().tolist()
# get a list of unique ID's
id_person = data_all['id'].unique().tolist()

#describe variable for information
for var in variables:
    print(var)
    data_var = data_all.loc[data_all.variable == var]
    print(data_var.describe().to_csv(index=False))

#remove outliers from selected variables
outlier_variables = [variables[i] for i in [4,7,8,9,12,14,15]]
#Loop through all selected variables
for var in outlier_variables:
    #Get data from selected variable
    df = data_all.loc[data_all.variable == var]
    #Find all values larger than 4 times standard deviation
    df = df[df.value-df.value.mean()>(4*df.value.std())]
    #Drop these values from the dataframe
    data_all = data_all.drop(df.index.values)

#remove values smaller than zero from selected variables
min_variables = [variables[i] for i in [7,9]]
#Loop through all selected variables
for var in min_variables:
    #Get data from selected variable
    df = data_all.loc[data_all.variable == var]
    #Find values smaller than zero
    df = df[df.value < 0]
    #Drops these values from the dataframe
    data_all = data_all.drop(df.index.values)

#save data to pickle file
data_all.to_pickle('preprocessed.pkl')
