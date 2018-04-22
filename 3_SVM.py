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

data_train = pd.read_pickle('filtered_data_train.pkl')
data_test = pd.read_pickle('filtered_data_test.pkl')

class helper_functions:
    #Scatterplot function
    def dataset_series(self, dataset):
        #Create new variable of call/sms combined
        dataset['call/sms'] = dataset['call'] + dataset['sms']
        dataset.pop('call')
        dataset.pop('sms')

        #Add ordinal dates
        dataset['ordinal'] = dataset.index.values

        # get a list of unique ID's
        id_person = dataset['id'].unique().tolist()

        # sort the dataframe by ID
        dataset.sort_values(by=['id'])
        dataset.set_index(keys=['id'], drop=False, inplace=True)

        #create names list for new dataframe
        total_names = []
        names = dataset.columns.values.tolist()
        names.append(names.pop(names.index('mood')))
        names.pop(names.index('id'))
        names.pop(names.index('ordinal'))
        for i in range(5):
            total_names.extend(names[0:-1])
            total_names.extend([names[-1]])
        total_names.extend(['mood_predict'])
        names.insert(0, 'ordinal')

        #create new dataframe for data
        set = pd.DataFrame(columns=total_names) # columns are the variable names

        for id in id_person:
            #Get data from 1 ID
            data_id = dataset.loc[dataset['id'] == id]
            data_id = data_id[names]
            #Get a list with unique sorted ordinal dates
            ordinal = sorted(data_id['ordinal'].unique().tolist())
            #create new list for serie
            set_serie = []

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
                    set_serie.append(week)

            set = set.append(pd.DataFrame(np.array(set_serie),columns=total_names), ignore_index=True)

        return set

help_func = helper_functions()
train_set = shuffle(help_func.dataset_series(data_train))
test_set  = shuffle(help_func.dataset_series(data_test))

array_train = train_set.values
X_train = array_train[:,:-1]
# X[np.argwhere(np.isnan(X))] = 0
Y_train = array_train[:,-1]

array_test = test_set.values
X_test = array_test[:,:-1]
# X[np.argwhere(np.isnan(X))] = 0
Y_test = array_test[:,-1]

# pca = PCA().fit(train_set.values)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.savefig('visualize/PCA.png')

pca = PCA(n_components=14)
features = pca.fit_transform(X_train)
features_analyse = pca.fit(X_train)

X_train = np.concatenate((X_train,features),axis=1)

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


from sklearn.externals import joblib
# joblib.dump(selector, 'selector.pkl')

selector = joblib.load('selector.pkl')
print('Optimal number of features :', selector.n_features_)
print('Best features :', selector.support_)

indexes = [i for i, x in enumerate(selector.support_) if x]

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (min value of MSE)")
# plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
# plt.show()


X_train1 = X_train[:,[indexes[0]]]
X_test1 = X_test[:,[indexes[0]]]

X_train2 = X_train[:,[indexes[1]]]
X_test2 = X_test[:,[indexes[1]]]

X_train3 = X_train[:,[indexes[2]]]
X_test3 = X_test[:,[indexes[2]]]

X_train = np.concatenate((X_train1,X_train2))
X_test = np.concatenate((X_test1, X_test2))

X_train = np.concatenate((X_train,X_train3))
X_test = np.concatenate((X_test, X_test3))


svr = SVR(C=1.0, epsilon=0.2)

svr.fit(X_train1, Y_train)

print('MSE')
print(mean_squared_error(svr.predict(X_test1), Y_test))
