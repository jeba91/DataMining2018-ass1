import pandas as pd
import copy
import matplotlib.pyplot as plt
from VisualizeDataset import VisualizeDataset
from pandas import DataFrame
import os

# Functions for data analysis and data cleaning/interpolation
class helper_functions:
    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

    def variable_per_participant(self, variable_name, dataframe, person_ids):
        dataframe.sort_values(by=['id'])
        # set the index to be this and don't drop
        dataframe.set_index(keys=['id'], drop=False, inplace=True)
        # get a list of id names
        # Total nummer of id's = 27 (some are missing!)
        #Get empty list for ids
        ids = []

        #Loop through al unique IDs and get Values
        #Then get values for certain ID and variable name
        for x in person_ids:
            #get only for one id
            data = dataframe.loc[dataset.id == x]
            #and the one value
            ids.append(data.loc[data.variable == variable_name])

        fig, axes = plt.subplots(6,5,sharex = True, sharey=True )

        #for subplots
        a = 0
        b = 0

        #create a subplot of the value against time for every ID
        for y in range(0,len(ids)):
            dir = 'pictures/' + person_ids[y] + '/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            savename2 = 'pictures/' + person_ids[y] + '/' + variable_name + '.png'
            fig2 = plt.figure()
            plt.plot(ids[y]['time'].values,ids[y]['value'].values)
            fig2.savefig(savename2)
            plt.close(fig2)

        #for subplots
        a = 0
        b = 0
        #save the plot
        for y in range(0,len(ids)):
            axes[a,b].plot(ids[y]['time'].values,ids[y]['value'].values)
            #axes[a,b].set_title(ids[y])
            a = a+1
            if a == 6:
                b = b+1
                a = 0
        savename = 'pictures/' + variable_name + '.png'
        fig.savefig(savename)

#load classes with functions needed
DataViz = VisualizeDataset()
help_func = helper_functions()

#create parses for formatted date data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')

#Load the dataset from CSV
try:
    dataset = pd.read_csv("dataset_mood_smartphone.csv", index_col=0, parse_dates=['time'], header=0, date_parser=dateparse)
except IOError as e:
    print('File not found!')
    raise e

# sort the dataframe
dataset.sort_values(by=['variable'])
# set the index to be this and don't drop
dataset.set_index(keys=['variable'], drop=False,inplace=True)

# get a list of unique variable names
variables = dataset['variable'].unique().tolist()
# get a list of unique ID's
id_person = dataset['id'].unique().tolist()


#Loop through all variable names from the dataset and perform interpolation
#Variable names are for example [mood, sms, call ...]
varias = []
for var in variables:
    #append the dataset with only values for one variable
    varias.append(dataset.loc[dataset.variable==var])

#loop through the list of variables values
#Count how much missing values we have for each variable;
#THIS ONE ONLY NEEDED FOR MISSING VALUE SEARCH
#for y in range(len(varias)):
    #print variable name
#    print(variables[y])
    #print amount of values
#    print(len(varias[y]))
    #print amount of NA values in variable value dataset
#    print(varias[y].isnull().sum())

#
interpol_dataset = help_func.impute_interpolate(copy.deepcopy(dataset), 'value')

# DataViz.plot_imputed_values(dataset, ['original', 'interpolation'], 'value', dataset['value'], imputed_interpolation_dataset['value'])
#print imputed_interpolation_dataset

#create images for every variable, saved in pictures
for name in variables:
    print(name)
    help_func.variable_per_participant(name, dataset, id_person)

interpol_dataset.to_pickle('interpolated')
