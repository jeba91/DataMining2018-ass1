import pandas as pd
import copy
import matplotlib.pyplot as plt
from VisualizeDataset import VisualizeDataset
from pandas import DataFrame
import os
import numpy as np
import math
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Functions for data analysis and data cleaning/interpolation
class helper_functions:
    # Interpolate the dataset based on previous/next values..
    def impute_interpolate(self, dataset, col):
        dataset[col] = dataset[col].interpolate()
        # And fill the initial data points if needed:
        dataset[col] = dataset[col].fillna(method='bfill')
        return dataset

#Function from the interwebs for NaNs
def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

#load classes with functions needed
DataViz = VisualizeDataset()
help_func = helper_functions()

#create parses for formatted date data
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')

#Load the dataset from CSV
try:
    data = pd.read_csv("dataset_mood_smartphone.csv", index_col=0, parse_dates=['time'], header=0, date_parser=dateparse)
except IOError as e:
    print('File not found!')
    raise e


# get a list of unique variable names
variables = data['variable'].unique().tolist()
# get a list of unique ID's
id_person = data['id'].unique().tolist()

#werkt niet echt?
#data = help_func.impute_interpolate(copy.deepcopy(data), 'value')

#TODO dit moet nog naar alle ID's maar nu testen op 1 ID
#Get one ID
data = data.loc[data['id'] == 'AS14.12']


# voeg dagen en maanden toe als kolommen voor makkelijk loopje
data['day'] = data['time'].dt.day
data['month'] = data['time'].dt.month

#Krijg een lijst met maanden als nummering
months = data['month'].unique().tolist()

# sort the dataframe
data.sort_values(by=['variable'])
# set the index to be this and don't drop
data.set_index(keys=['variable'], drop=False, inplace=True)

#lege lijst voor totale trainings data array
total_data = []

#TODO dit is nu een score van T-1 voorspelt T, nemen we 7 dagen en dus 7xN aantal variabelen
# of nemen we gemiddelde score van T-1:6 en dat als voorspellend gebruiken (zie opdracht)

#TODO checken of nummering van dagen wel klopt, index 1,2,3 wel opeenvolgende dagen enzo

#Loop door alle maanden in de data
for m in months:
    #Pak alleen de data voor deze maand
    data_month = data.loc[data.month == m]
    #Voor deze maand, alle dagen met data
    days = data['day'].unique().tolist()
    #loop door al deze dagen
    for d in days:
        #een lijst voor het gemiddelde per dag
        means_day = []
        #Pak per dag alle data van die dag
        data_day = data_month.loc[data_month.day == d]
        #loop door alle variabele
        for v in variables:
            #pak het gemiddelde van 1 variabele van 1 dag
            data_v = data_day.loc[data_day.variable == v]
            #Voeg dit gemiddelde aan de lijst toe
            means_day.append(np.mean(data_v['value'].values))
        #voeg de lijst met gemiddeldes van alle variabele voor 1 dag toe
        total_data.append(means_day)


#zet totale data om in een numpy array (Handiger)
total_data = np.array(total_data)

#TODO als je alles hieronder comment print hij alle rijen aan data met NaNs en zie je mooi welke dagen wel en niet bruikbaar zijn
#print totale voor analyse
for i in range(np.shape(total_data)[0]):
    print(i, total_data[i])

# linear interpolation of NaNs
#transpose de matrix zodat elke rij een variabele is
total_data = total_data.transpose()

#TODO kijken of dit beter kan want nu zitten er -6 values tussen voor bijvoorbeeld een range van -2 tot 2
#loop over elke rij en interpoleer de NaN values
for i in range(np.shape(total_data)[0]):
    y = total_data[i,:]
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    total_data[i,:] = y

#transpose weer terug
total_data = total_data.transpose()

#print date met NaNs weg
for i in range(np.shape(total_data)[0]):
    print(i, total_data[i])

#TODO deze dagen niet handmatig uitkiezen
#deze dagen heb ik handmatig uitgekozen omdat ze het beste leken
X = np.array(total_data[31:60,:])
Y = np.array(total_data[32:61,0])
Y = np.round(Y,0)
X = np.add(X,10)

#om te checken als je wilt
# print(X)
# print(Y)
# print(variables)

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
numpy.set_printoptions(precision=3)
scores = fit.scores_

#voor elke variabele print de score
for s in range(len(scores)):
    print(variables[s], '=', scores[s])
features = fit.transform(X)

#check de link in app, iets met score per feature ofzo
# summarize selected features
print(features[0:5,:])
