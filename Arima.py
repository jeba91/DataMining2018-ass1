from pandas import read_csv
from pandas import datetime
import pandas as pd
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S.%f')

try:
    dataset = pd.read_csv("dataset_mood_smartphone.csv", index_col=0, parse_dates=['time'], header=0, date_parser=dateparse)
except IOError as e:
    print('File not found!')
    raise e

# new_dataset = dataset.loc[:, ['variable','value']]
# print new_dataset

df = pd.DataFrame(data=dataset, columns=['variable', 'value'])
print df

# autocorrelation_plot(new_dataset)
# pyplot.show()

