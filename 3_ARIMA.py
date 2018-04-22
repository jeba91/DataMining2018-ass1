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
# from statsmodels.tsa.stattools import acf, pacf
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

# # sort the dataframe by variables
# data_train.sort_values(by=['id'])
# data_train.set_index(keys=['id'], drop=False, inplace=True)
# data_test.sort_values(by=['id'])
# data_test.set_index(keys=['id'], drop=False, inplace=True)

train_set = data_train[['id','ordinal','mood']]
test_set = data_test[['id','ordinal','mood']]

print(train_set)

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



from sklearn.metrics import mean_squared_error

for id in id_person:
	fig = plt.figure()
	train_id = train_set.loc[train_set['id'] == id]
	test_id =  test_set.loc[train_set['id'] == id]
	train = train_id['mood'].values.tolist()
	X = 
	size = int(len(X) * 0.66)
	train, test = X[0:size], X[size:len(X)]
	history = [x for x in train]
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=(1,1,0))
		model_fit = model.fit(disp=0)
		output = model_fit.forecast()
		yhat = output[0]
		predictions.append(yhat)
		obs = test[t]
		history.append(obs)
		print('predicted=%f, expected=%f' % (yhat, obs))
	error = mean_squared_error(test, predictions)
	print('Test MSE: %.3f' % error)
	# plot
	pyplot.plot(test)
	pyplot.plot(predictions, color='red')
	pyplot.savefig('arima/' + id + '.png')

# X = data_id['mood']
# Y = range(len(data_id['mood']))
#
# plt.plot(Y, X)
# plt.ylabel('Mood')
# plt.show()
#
# lag_plot(np.log(data_id['mood']))
# plt.show()
#
# autocorrelation_plot(np.log(data_id['mood']))
# pyplot.show()

#
# # fit model
# model = ARIMA(np.log(data_id['mood']), order=(7, 1, 0))
# model_fit = model.fit(disp=0)
# print(model_fit.summary())

# #plot residual errors
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# pyplot.show()
# residuals.plot(kind='kde')
# pyplot.show()
# print(residuals.describe())
# pyplot.show()
# print(data_id)
