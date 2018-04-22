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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from datetime import datetime
from tabulate import tabulate
import numpy

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor

#Load in filtered data
data_all = pd.read_pickle('filtered_data_train.pkl')

#Create new variable of call/sms combined
data_all['call/sms'] = data_all['call'] + data_all['sms']
data_all.pop('call')
data_all.pop('sms')

#Add ordinal dates
data_all['ordinal'] = data_all.index.values

# get a list of unique ID's
id_person = data_all['id'].unique().tolist()

# sort the dataframe by variables
data_all.sort_values(by=['id'])
# set the index to be this and don't drop
data_all.set_index(keys=['id'], drop=False, inplace=True)


names = data_all.columns.values.tolist()
names.append(names.pop(names.index('mood')))
names.pop(names.index('id'))
names.pop(names.index('ordinal'))

total_names = []
for i in range(5):
    total_names.extend(names[0:-1])
    total_names.extend([names[-1]])
total_names.extend(['mood_predict'])

names.insert(0, 'ordinal')

train_set = pd.DataFrame(columns=total_names) # columns are the variable names


for id in id_person:
    #Get data from 1 ID
    data_id = data_all.loc[data_all['id'] == id]
    data_id = data_id[names]
    #Get a list with unique sorted ordinal dates
    ordinal = sorted(data_id['ordinal'].unique().tolist())

    train_set_week = []

    for j in ordinal:
        days = [5,4,3,2,1,0]
        past = [j-i for i in days]
        past_in_ordinal = [x for x in past if x in ordinal]
        if len(past_in_ordinal) == 6:
            week = []
            for p in past:
                if p != past[-1]:
                    data_p = data_id.loc[data_id['ordinal'] == p].values.tolist()
                    week.extend(data_p[0][1:-1])
                    week.extend([data_p[0][-1]])
                else:
                    week.extend([data_p[0][-1]])
            train_set_week.append(week)

    train_ord = pd.DataFrame(np.array(train_set_week),
                             columns=total_names) # columns are the variable names

    train_set = train_set.append(train_ord, ignore_index=True)


# pca = PCA().fit(train_set.values)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.savefig('visualize/PCA.png')

train_set = shuffle(train_set)
# training_set, testing_set = train_test_split(train_set, test_size=0.2)

array = train_set.values
X = array[:,:-1]
# X[np.argwhere(np.isnan(X))] = 0
Y = array[:,-1]

print(X)
print(Y)

pca = PCA(n_components=14)
features = pca.fit_transform(X)
features_analyse = pca.fit(X)

X = np.concatenate((X,features),axis=1)

# # Build a forest and compute the feature importances
# forest = ExtraTreesRegressor(n_estimators=250,
#                               random_state=0)
#
# forest.fit(X, Y)
#
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()


# estimator = SVR(kernel="rbf")
# selector = RFECV(estimator, step=1, cv=3, verbose=1, scoring='neg_mean_squared_error')
# selector = selector.fit(X, Y)
#

#
from sklearn.externals import joblib
# joblib.dump(selector, 'selector.pkl')

selector = joblib.load('selector.pkl')
print('Optimal number of features :', selector.n_features_)
print('Best features :', selector.support_)
print(selector.grid_scores_)

print(len(selector.support_))
print(len(total_names))

indexes = [i for i, x in enumerate(selector.support_) if x]

print([total_names[i] for i in indexes])


# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (min value of MSE)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()

X = [X[i] for i in indexes]
Y = [Y[i] for i in indexes]

# #TRAIN SVM
# array = train_set.values
# X_test = array[:,indexes]
# # X_test[np.argwhere(np.isnan(X))] = 0
# Y_test = array[:,-1]

svr = SVR(C=1.0, epsilon=0.2)

svr.fit(X, Y)

print(mean_squared_error(svr.predict(X), Y))
