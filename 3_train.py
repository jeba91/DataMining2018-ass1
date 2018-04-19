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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


data_all = pd.read_pickle('train_set.pkl')
data_all['ordinal'] = data_all.index.values

# get a list of unique ID's
id_person = data_all['id'].unique().tolist()

# sort the dataframe by variables
data_all.sort_values(by=['id'])
# set the index to be this and don't drop
data_all.set_index(keys=['id'], drop=False, inplace=True)

names = data_all.columns.values.tolist()
total_names = []
for i in range(5):
    total_names.extend(names[0:15])
total_names.extend(['mood_predict'])

train_set = pd.DataFrame(columns=total_names) # columns are the variable names

for id in id_person:
    #Get data from 1 ID
    data_id = data_all.loc[data_all['id'] == id]

    #Get a list with unique sorted ordinal dates
    ordinal = sorted(data_id['ordinal'].unique().tolist())

    train_set_week = []

    for j in ordinal:
        days = [5,4,3,2,1,0]
        past = [j-i for i in days]
        past_in_ordinal = [x for x in past if x in ordinal]
        if len(past_in_ordinal) == 6:
            week = []
            for p in past:
                if p != past[-1]:
                    data_p = data_id.loc[data_id['ordinal'] == p].values.tolist()
                    week.extend(data_p[0][0:15])
                else:
                    week.extend([data_p[0][0]])
            train_set_week.append(week)

    train_ord = pd.DataFrame(np.array(train_set_week),
                             columns=total_names) # columns are the variable names

    train_set = train_set.append(train_ord, ignore_index=True)

train_set = shuffle(train_set)
training_set, testing_set = train_test_split(train_set, test_size=0.2)

print(training_set)
