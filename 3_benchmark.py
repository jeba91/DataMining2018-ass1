import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pandas
import pandas as pd
import datetime as dt
import seaborn as sns
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from tabulate import tabulate
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
import numpy
from pandas.tools.plotting import lag_plot
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import acf, pacf
from sklearn.metrics import mean_squared_error

import pandas
import pandas as pd

#Load in filtered data
data_train = pd.read_pickle('filtered_data_train.pkl')
data_test = pd.read_pickle('filtered_data_test.pkl')

#Add ordinal dates
data_train['ordinal'] = data_train.index.values
data_test['ordinal'] = data_test.index.values

# get a list of unique ID's
id_person = data_train['id'].unique().tolist()

#Get train set with 3 columns
train_set = data_train[['id','ordinal','mood']]
test_set = data_test[['id','ordinal','mood']]

print(train_set)
total_mse = 0

#create benchmark values
for id in id_person:
    train_id = train_set.loc[train_set['id'] == id]
    test_id =  test_set.loc[train_set['id'] == id]
    train_id = pd.concat([train_id,test_id])
    mood = train_id['mood'].values.tolist()
    y_pred = mood[:-1]
    y_true = mood[1:]
    error = mean_squared_error(y_pred, y_true)
    total_mse += error
    print(id, 'Test MSE: %.3f' % error)

#Create MSE
print('mean', total_mse/len(id_person))
