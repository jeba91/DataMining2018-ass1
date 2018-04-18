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
from datetime import datetime
import numpy

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


data_all = pd.read_pickle('train_set.pkl')
data_all['ordinal'] = data_all.index.values

#Get data from 1 ID
data_id = data_all.loc[data_all['id'] == 'AS14.01']

# sort the dataframe by variables
data_id.sort_values(by=['id'])
# set the index to be this and don't drop
data_id.set_index(keys=['id'], drop=False, inplace=True)

#Get a list with unique sorted ordinal dates
ordinal = sorted(data_id['ordinal'].unique().tolist())

print(data_id)

train_set_week = pd.DataFrame()

for j in ordinal:
    days = [5,4,3,2,1,0]
    past = [j-i for i in days]
    past_in_ordinal = [x for x in past if x in ordinal]
    if len(past_in_ordinal) == 6:
        week = []
        for p in past:
            week.append(data_id.loc[data_id['ordinal'] == p])
        train_set_week.append(week)

print(train_set_week)

# #Loop through all unique ordinals
# for ord in ordinal:
#     #Get only data for one ordinal
#     data_ord = data_id.loc[data_id.ordinal == ord].values
#     print(data_ord)
