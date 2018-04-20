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
from datetime import datetime as dt
import numpy

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Functions for data analysis and data cleaning/interpolation
class helper_functions:
    #Function to find data from ID formatted
    def create_person_df(self, dataset, id, variables):
        #variables to sum instead of mean
        sum_var = [variables[i] for i in [5,6]]
        #Get data from 1 ID
        data_id = dataset.loc[dataset['id'] == id]

        #Convert dates to ordinal system
        data_id['ordinal'] = [x.toordinal() for x in data_id['time'].dt.date.tolist()]

        #Get a list with unique sorted ordinal dates
        ordinal = sorted(data_id['ordinal'].unique().tolist())

        # sort the dataframe by variables
        data_id.sort_values(by=['variable'])
        # set the index to be this and don't drop
        data_id.set_index(keys=['variable'], drop=False, inplace=True)

        #Empty list for total data
        total_data = []

        #Loop through all unique ordinals
        for ord in ordinal:
            #Get only data for one ordinal
            data_ord = data_id.loc[data_id.ordinal == ord]
            means_ord = []
            #loop through all variables

            for v in variables:
                #Get the mean for every value of one day
                data_v = data_ord.loc[data_ord.variable == v]
                #Add mean to list
                if v in sum_var:
                    means_ord.append(np.sum(data_v['value'].values))
                else:
                    means_ord.append(np.mean(data_v['value'].values))
            #Add all means as list to total data
            total_data.append(means_ord)

        #Create dataframe with data per day
        person = pd.DataFrame(data=np.array(total_data),        # values
                              index=ordinal,                    # Index are the ordinals
                              columns=variables) # columns are the variable names

        return person

    #Function to find serie of values with <3 NaNs
    def find_series(self, dataset):
        nan = 0
        start = 0
        series = []

        max_index = max(list(dataset.index.values))

        for index,row in dataset.iterrows():
            if index == max_index and start != 0:
                series.append([start,index,index-start])
            elif math.isnan(row['mood']):
                nan += 1
                if nan > 3 and start != 0:
                    series.append([start,index,index-start])
                    start = 0
            else:
                nan = 0
                if start == 0:
                    start = index

        return series


help_func = helper_functions()
data_all = pd.read_pickle('preprocessed.pkl')

# get a list of unique variable names
variables = data_all['variable'].unique().tolist()
variables = [variables[i] for i in [0,1,2,3,4,5,6,8,9,10,11,12,14,15,18]]

# get a list of unique ID's
id_person = data_all['id'].unique().tolist()

total_series = []

#create series from each ID and total list
for id in id_person:
    df_person = help_func.create_person_df(data_all, id, variables)
    series = help_func.find_series(df_person)
    for s in series:
        total_series.append([id, s])


training_data = pd.DataFrame(columns=variables) # columns are the variable names

for serie in total_series:
    df_person = help_func.create_person_df(data_all, serie[0], variables)
    df_person = df_person.loc[range(serie[1][0],serie[1][1],1)]
    df_person[variables[0:3]] = df_person[variables[0:3]].interpolate()
    df_person[variables[3:]] = df_person[variables[3:]].fillna(0)
    df_person['id'] = serie[0]
    training_data = pd.concat([training_data, df_person])


for ser in total_series:
    if ser[1][2] < 8:
        df_person = help_func.create_person_df(data_all, serie[0], variables)
        df_person = df_person.loc[range(serie[1][0],serie[1][1],1)]
        print(df_person)


# #for subplots
# fig, axes = plt.subplots(4,5, sharey=True )
# a = 0
# b = 0
#
# #scatter plot totalbsmtsf/saleprice
# for var in variables:
#     data = pd.concat([training_data['mood'], training_data[var]], axis=1)
#     data.plot.scatter(ax=axes[a,b], x=var, y='mood'); axes[a,b].set_title(var)
#     a = a+1
#     if a == 4:
#         b = b+1
#         a = 0
#
# plt.show()

# #correlation matrix
# corrmat = training_data.corr()
# f, ax = plt.subplots(figsize=(12, 9))
# sns.heatmap(corrmat, vmax=.8, square=True);
# plt.xticks(rotation='vertical')
# plt.yticks(rotation='horizontal')
# plt.show()


#save data to pickle file
training_data.to_pickle('train_set.pkl')
