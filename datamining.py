import pandas as pd
from VisualizeDataset import VisualizeDataset
import copy
import matplotlib.pyplot as plt
from pandas import DataFrame

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')

try:
    dataset = pd.read_csv("dataset_mood_smartphone.csv", index_col=0, parse_dates=['time'], header=0, date_parser=dateparse)
except IOError as e:
    print('File not found!')
    raise e

# sort the dataframe
dataset.sort_values(by=['variable'])
# set the index to be this and don't drop
dataset.set_index(keys=['variable'], drop=False,inplace=True)
# get a list of names
variables = dataset['variable'].unique().tolist()
# now we can perform a lookup on a 'view' of the dataframe


varias = []

#Loop through all variable names from the dataset and perform interpolation
#Variable names are for example [mood, sms, call ...]
for x in variables:
    #append the dataset with only values for one variable
    varias.append(dataset.loc[dataset.variable==x])

#loop through the list of variables values
#Count how much missing values we have for each variable;
#THIS ONE ONLY NEEDED FOR MISSING VALUE SEARCH
#for y in range(0, len(varias)):
    #print variable name
    #print(variables[y])
    #print amount of values
    #print(len(varias[y]))
    #print amount of NA values in variable value dataset
    #print(varias[y].isnull().sum())


# Missing Values
class ImputationMissingValues:
    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset
#
DataViz = VisualizeDataset()
MisVal = ImputationMissingValues()
imputed_interpolation_dataset = MisVal.impute_interpolate(copy.deepcopy(dataset), 'value')

# DataViz.plot_imputed_values(dataset, ['original', 'interpolation'], 'value', dataset['value'], imputed_interpolation_dataset['value'])
#print imputed_interpolation_dataset

def mood_per_participant(variable_name):
    dataset.sort_values(by=['id'])
    # set the index to be this and don't drop
    dataset.set_index(keys=['id'], drop=False, inplace=True)
    # get a list of id names
    # Total nummer of id's = 27 (some are missing!)
    id_person = dataset['id'].unique().tolist()

    #Get empty list for ids
    ids = []

    #Loop through al unique IDs and get Values
    #Then get values for certain ID and variable name
    for x in id_person:
        #get only for one id
        data = dataset.loc[dataset.id == x]
        #and the one value
        ids.append(data.loc[data.variable == variable_name])

    fig, axes = plt.subplots(6,5,sharex = True, sharey=True )

    #for subplots
    a = 0
    b = 0

    #create a subplot of the value against time for every ID
    for y in range(0,len(ids)):
        #print(a,b)
        axes[a,b].plot(ids[y]['time'].values,ids[y]['value'].values)
        a = a+1
        if a == 6:
            b = b+1
            a = 0

    #save the plot
    savename = 'pictures/' + variable_name + '.png'
    fig.savefig(savename)
    #plt.show()

#create images for every variable, saved in pictures
for name in variables:
    print(name)
    mood_per_participant(name)
