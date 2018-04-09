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

for x in variables:
    varias.append(dataset.loc[dataset.variable==x])

# Count how much missing values we have for each variable;
for y in range(0, len(varias)):
    print(variables[y])
    print(len(varias[y]))
    print(varias[y].isnull().sum())


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

def mood_per_participant():
    dataset.sort_values(by=['id'])
    # set the index to be this and don't drop
    dataset.set_index(keys=['id'], drop=False, inplace=True)
    # get a list of names
    id_person = dataset['id'].unique().tolist()
    print(id_person)
    ids = []



    print(dataset.id)


    for x in id_person:
        data = dataset.loc[dataset.id == x]
        ids.append(data.loc[data.variable == 'mood'])

    #for y in ids:
    print(ids[0])
    print(ids[0]['time'].values)
    print(ids[0]['value'].values)

    fig, axes = plt.subplots(6,5,sharex = True, sharey=True )

    a = 0
    b = 0

    print(len(ids))

    for y in range(0,len(ids)):
        print(a,b)
        axes[a,b].plot(ids[y]['time'].values,ids[y]['value'].values)
        a = a+1
        if a == 6:
            b = b+1
            a = 0

    plt.show()

mood_per_participant()