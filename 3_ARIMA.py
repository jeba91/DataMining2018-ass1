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
from statsmodels.tsa.arima_model import ARIMA
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from tabulate import tabulate
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
import numpy
from pandas.tools.plotting import lag_plot
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.stattools import acf, pacf

import pandas
import pandas as pd

#Load in filtered data
data_train = pd.read_pickle('filtered_data_train.pkl')
data_test = pd.read_pickle('filtered_data_test.pkl')

#Add ordinal dates
data_train['ordinal'] = data_train.index.values
data_test['ordinal'] = data_test.index.values

# get a list of unique ID's
id_person = data_train['id'].unique().tolist()

#Get only n columns
train_set = data_train[['id','ordinal','mood']]
test_set = data_test[['id','ordinal','mood']]

print(train_set)

#Get autocorrelation_plot for every ID
for id in id_person:
	fig = plt.figure()
	train_id = train_set.loc[train_set['id'] == id]
	test_id =  test_set.loc[train_set['id'] == id]

	lag_acf = acf((np.log(train_id['mood'])), nlags=20)
	lag_pacf = pacf((np.log(train_id['mood'])), nlags=20, method='ols')

	plt.subplot(121)
	plt.plot(lag_acf)
	plt.axhline(y=0,linestyle='--',color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(np.log(train_id['mood']))),linestyle='--',color='gray')
	plt.axhline(y=1.96/np.sqrt(len(np.log(train_id['mood']))),linestyle='--',color='gray')
	plt.title('Autocorrelation Function')

	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0,linestyle='--',color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(np.log(train_id['mood']))),linestyle='--',color='gray')
	plt.axhline(y=1.96/np.sqrt(len(np.log(train_id['mood']))),linestyle='--',color='gray')
	plt.title('Partial Autocorrelation Function')
	plt.tight_layout()
	plt.savefig('acfpacf/' + id + '.png')

#Set different values for different ID's
norm = 2
p = [norm for i in range(len(id_person))]
p[3]  = 4
p[12] = 3
p[16] = 3
p[23] = 1

from sklearn.metrics import mean_squared_error

total_mse = 0

#Train ARIMA on every person and get results
for id in id_person:
	p_id = p[id_person.index(id)]
	train_id = train_set.loc[train_set['id'] == id]
	test_id =  test_set.loc[test_set['id'] == id]
	train = train_id['mood'].values.tolist()
	test = test_id['mood'].values.tolist()
	predictions = list()
	history = [x for x in train]
	for t in range(len(test)):
		model = ARIMA(history, order=(p_id,1,0))
		model_fit = model.fit(disp=0)
		fore = model_fit.forecast()
		yhat = fore[0][0]
		obs  = test[t]
		predictions.append(yhat)
		history.append(obs)
		# print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	total_mse += error
	print(id, 'Test MSE: %.3f' % error)
	# plot
	plt.figure()
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.savefig('arima/' + id + '.png')


print('mean', total_mse/len(id_person))
