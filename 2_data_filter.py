import copy
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.random as random
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
        #Number of consecutive NaN values
        nan = 0
        #Start value
        start = 0
        #Series list
        series = []
        #last ordinal
        max_index = max(list(dataset.index.values))
        #loop through all rows of data
        for index,row in dataset.iterrows():
            #Last index and part of a serie, save serie
            if index == max_index and start != 0:
                series.append([start,index,index-start])
            #If mood NaN, count how much until value
            elif math.isnan(row['mood']):
                nan += 1
                #If more than 3 NaN, save serie
                if nan > 3 and start != 0:
                    series.append([start,index,index-start])
                    start = 0
            #Else, start serie, or just keep serie going
            else:
                nan = 0
                if start == 0:
                    start = index
        #Return list of series
        return series

# Functions for data analysis and data cleaning/interpolation
class visualize_functions:
    #Scatterplot function
    def scatterplot(self, dataset, variables):
        #for subplots
        fig, axes = plt.subplots(3,3, sharey=True )
        a = 0
        b = 0

        #scatter plot totalbsmtsf/saleprice
        for var in variables:
            if var == 'mood':
                break
            data = pd.concat([dataset['mood'], dataset[var]], axis=1)
            data.plot.scatter(ax=axes[a,b], x=var, y='mood'); axes[a,b].set_title(var)
            a = a+1
            if a == 3:
                b = b+1
                a = 0

        fig.savefig('visualize/scatterplot.png')

    #QQ plot function
    def QQplot(self, dataset, variables):
        for var in variables:
            data_var = dataset[var].values.flatten()
            data_var.sort()
            norm  = random.normal(0,2,len(data_var))
            norm.sort()
            fig = plt.figure(figsize=(12,8),facecolor='1.0')
            plt.plot(norm,data_var,"o")
            z = np.polyfit(norm,data_var, 1)
            p = np.poly1d(z)
            plt.plot(norm,p(norm),"k--", linewidth=2)
            plt.title("Normal Q-Q plot "+var, size=28)
            plt.xlabel("Theoretical quantiles", size=24)
            plt.ylabel("Expreimental quantiles", size=24)
            plt.tick_params(labelsize=16)
            fig.savefig('visualize/'+var+'.png')

    #heatmap function
    def heatmap_corr(self, dataset):
        #UNCOMMENT FOR PLOTTING CORRELATION MATRIX
        corrmat = dataset.corr()
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True);
        plt.xticks(rotation='vertical')
        plt.yticks(rotation='horizontal')
        fig.savefig('visualize/heatmap.png')

#Load classes
help_func = helper_functions()
vis_func = visualize_functions()

#Read in preprocessed data
data_all = pd.read_pickle('preprocessed.pkl')

# get a list of unique variable names
variables = data_all['variable'].unique().tolist()
# get the variables selected for analysis
variables = [variables[i] for i in [0,1,2,3,4,5,6,8,9,10,11,12,14,15,18]]

#Remove sparse variables
variables = [variables[i] for i in [0,1,2,3,4,5,6,7,8,12]]


# get a list of unique ID's
id_person = data_all['id'].unique().tolist()

#Get an empty list for time series
total_series = []

#create series from each ID and total list
for id in id_person:
    #Get dataframe of one ID
    df_person = help_func.create_person_df(data_all, id, variables)
    #Find series of this ID
    series = help_func.find_series(df_person)
    #Append serie for serie with ID number
    for s in series:
        total_series.append([id, s])

#Create new dataframe for filtered data
filtered_data = pd.DataFrame(columns=variables)

#remove small series
total_series = [x for x in total_series if x[1][2] > 15]

#Loop through all series and extract data of serie
for serie in total_series:
    #Get dataframe of one ID
    df_person = help_func.create_person_df(data_all, serie[0], variables)
    #Get data of dataframe of one serie
    df_person = df_person.loc[range(serie[1][0],serie[1][1],1)]
    #Interpolate mood, valence and arousal.
    df_person[variables[0:3]] = df_person[variables[0:3]].interpolate()
    #Fill NaN of time based data
    df_person[variables[3:]] = df_person[variables[3:]].fillna(0)
    #add column with ID to data
    df_person['id'] = serie[0]
    #Concat total filtered data with person fitlered data
    filtered_data = pd.concat([filtered_data, df_person])


#visualize data
vis_func.scatterplot(filtered_data,variables)
vis_func.QQplot(filtered_data,variables)
vis_func.heatmap_corr(filtered_data)

#save data to pickle file
filtered_data.to_pickle('filtered_data.pkl')
