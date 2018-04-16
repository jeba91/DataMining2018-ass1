import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import math
import pandas
import pandas as pd
import datetime as dt
from pandas import DataFrame
from datetime import datetime
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Functions for data analysis and data cleaning/interpolation
class helper_functions:
    def create_person_df(self, dataset, id, variables):
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
                means_ord.append(np.mean(data_v['value'].values))
            #Add all means as list to total data
            total_data.append(means_ord)

        #Create dataframe with data per day
        person = pd.DataFrame(data=np.array(total_data),    # values
                              index=ordinal,                # Index are the ordinals
                              columns=variables)            # columns are the variable names

        return person



#create parses for formatted date data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')
help_func = helper_functions()

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

df_person = help_func.create_person_df(data_all, 'AS14.31', variables)

df_person['mood'] = df_person.apply(lambda x: x['mood'] if np.isnan(x['mood']) is False else 0, axis=1)

print(df_person['mood'])
