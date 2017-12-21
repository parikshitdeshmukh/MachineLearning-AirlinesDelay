from __future__ import division

#
# Naive Bayes.py
# Author: Parikshit Sunil Deshmukh
# Description: This code  reads the dataset into pandas dataframes, builds a Naive Bayes Classifier, predicts labels for a subset of data. It also calculates metrics such as precision/recall/accuracy and F-Score after classification. The output is dumped in pickle files which are used later for visualization
#

import pickle
import sklearn
from sklearn.naive_bayes import *
import pandas as pd
import numpy as np
from sklearn import *
import os
from sklearn.metrics import *
from sklearn import metrics, preprocessing
from sklearn import svm, naive_bayes, neighbors, tree
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import pandas as pd

#
# Function: createPickle()
# Description: This function will create a pickle file.
# Input: data structure that you want to pickle
# Output: a pickle file for the data structure. The file is stored in the
# same path the code is running from
#


TRAINING_LINE_NUMBER = 5  # Number of lines to be read from input files
# List of years for training and testing
YEARS = ['2003','2004','2005','2006','2007','2008']
INPUT_FILE_PATH = "/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/"  # Unix pathSKIP_FIRST_LINE = True  # To skip the first line, as its the header

# Creating the master data frame from all years.
master = []
#
# print("Reading into Pandas frame...")
# try:
#     for year in YEARS:
#         path = os.path.join(INPUT_FILE_PATH, '%d.csv' % int(year))
#         print("\n", path)
#         dfPart = pd.read_csv(
#             path, nrows=TRAINING_LINE_NUMBER, skiprows=0, usecols=[
#                 u'Year',
#                 u'Month',
#                 u'DayofMonth',
#                 u'DayOfWeek',
#                 u'UniqueCarrier',
#                 u'DepTime',
#                 u'TailNum',
#                 u'Origin',
#                 u'Dest',
#                 u'DepDelay',
#                 # u'ArrDelay',
#                 u'Cancelled',
#                 #                 u'ArrTime',
#                 #                 u'ArrDelay',
#                 #                 u'Distance'
#             ])
#         print(len(dfPart))
#         # Removing cancelled flights from each year
#         dfPart = dfPart[dfPart['Cancelled'] == 0]
#         rows = np.random.choice(np.random.permutation(dfPart.index.values), len(dfPart) // 3, replace=False)  # 33% sampling of training data
#         print(rows)
#         sampled_dfPart = dfPart.ix[rows]
#         sampled_dfPart = dfPart
#         master.append(sampled_dfPart)
# except Exception as e:
#     print("Supplemental Data Import failed", e)



def createPickle(data, filename):
    with open(filename, 'wb') as f:
            pickle.dump(data, f)
    print("Pickled", filename)


def typeconvert(findata):
    dfMaster = pd.DataFrame(np.array(findata))

    dfMaster.fillna(0, inplace=True)
    dfMaster[0] = dfMaster[0].astype('int')
    dfMaster[1] = dfMaster[1].astype('int')
    dfMaster[2] = dfMaster[2].astype('int')
    dfMaster[3] = dfMaster[3].astype('int')
    dfMaster[4] = dfMaster[4].astype('int')
    dfMaster[5] = dfMaster[5].astype('int')
    dfMaster[6] = dfMaster[6].astype('int')
    dfMaster[7] = dfMaster[7].astype('int')
    dfMaster[8] = dfMaster[8].astype('int')
    dfMaster[9] = dfMaster[9].astype('int')

    ##findata = dfMaster.values
    return dfMaster



key_data= np.genfromtxt("/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/Secret_Keys/Key_data.txt", delimiter=",", dtype=str)
pic = ['Data-2003-2000000', 'Data-2004-2000000', 'Data-2005-2000000', 'Data-2006-2000000','Data-2007-2000000','Data-2008-2000000' ]
conn = S3Connection(key_data[0], key_data[1])
bucket = conn.get_bucket('sdmairlines')
k = Key(bucket)


##Training the model
try:
    for el in pic:
        k.key = el
        k.get_contents_to_filename(el)
        findata = pickle.load(open(el, 'rb'))
        findata = typeconvert(findata)
        rows = np.random.choice(np.random.permutation(findata.index.values), len(findata) // 3, replace=False)  # 33% sampling of training data
        #cols = list(df.loc[:,'0':'6']) +
        sampled_dfPart = findata.ix[rows]
        ##sampled_dfPart = findata
        master.append(sampled_dfPart)
except Exception as e:
    print("Supplemental Data Import failed", e)

# Building the master frame by concating it for all years
df = pd.concat(master, ignore_index=True)
master = []
dfPart = []

# Since we dont have a classification label in the data, we are creating
# one. Threshold of 5mins was chosen.

#
# Function: getTailNum()
# Description: This function will convert the input categorical value to corresponding numeric key.
# Input: categorical value you want to convert
# Output: a numeric value corresponding to the value passed. It uses the list created previously for lookup.
#


# Building classifier
print("Begin cross validation...")

# Choosing features for classifier
features = df.columns[[0,1,2,3,4,5,6,8,9]]

# Creating lists for storing results for cross validation.
accuracy = {}
results = {}
matrix = {}
prec = {}
recall = {}




for year in YEARS:
    print("Testing on - ", year)
    train = df[df[0] != int(year)]  # Test on 1year, train on other 7years
    test = df[df[0] == int(year)]
    # test = test[test['Origin'].isin([Origin['OAK'], Origin['SFO']])]
    print(len(train), len(test))
    rows = np.random.choice(np.random.permutation(
                            test.index.values), len(test) // 2, replace=False)  # 50% sampling of test data to avoid memory errors faced.
    # print rows
    sampled_test = test.ix[rows]
    sampled_test = test
    # Putting the last column of Training data into a list
    trainTargets = np.array(train[7]).astype(int)

    # Putting the last column of Testing data into a list
    testTargets = np.array(sampled_test[7]).astype(int)
    print("Train length - ", len(train), "Test length -  ", len(sampled_test))
    print(train[0])
    print(test[0])
    print("Model fitting and prediction started...")
    # Building the classifier and fitting the train data
    gnb = GaussianNB()
    y_gnb = gnb.fit(train[features], trainTargets).predict(
        sampled_test[features])
    # Storing results in a new colum in the dataframe.
    sampled_test['pred_label'] = y_gnb
    print("Classification completed.")
    # Creating pickle files with the classifier and the results of classifier
    createPickle(gnb, INPUT_FILE_PATH + "classifier_" + year + ".pkl")
    createPickle(y_gnb, INPUT_FILE_PATH + "label_" + year + ".pkl")
    sampled_test.to_csv(
        INPUT_FILE_PATH + "\dfTest" + year + ".csv", index=False)
# Calculating metrics using sklearn metrics functions
    print("\nCalculating metrcs...")
    accuracy[int(year)] = accuracy_score(sampled_test[7], y_gnb)
    print("Accuracy score - ", accuracy[int(year)], "Year", year)
    prec[int(year)] = precision_score(
        sampled_test[7], y_gnb, average='micro')
    print("Precision Score - ", prec[int(year)])
    recall[int(year)] = recall_score(
        sampled_test[7], y_gnb, average='micro')
    print("Recall Score - ", recall[int(year)])
    print("Confusion matrix")
    matrix[int(year)] = metrics.confusion_matrix(
        sampled_test[7], y_gnb)
    print(matrix[int(year)])
    results[int(year)] = precision_recall_fscore_support(
        sampled_test[7], y_gnb, average='micro')
    print("Precision, recall, F-Score, Support - ", results[int(year)])
    print("Classification report")
    print(classification_report(np.array(sampled_test[7]), y_gnb,target_names=['Class0', 'Class1']))
    train = []
    test = []

print("Accuracy\n", accuracy)
print("\nPrecision\n", prec)
print("\nRecall\n", recall)
print("\nMetrics\n", results)
print("\nMatrix\n", matrix)

# # Finding mean of metrics
# print("\nMean Cross validation Precision score", np.mean(pd.Series(prec)))
# print("\nMean Cross validation Recall score", np.mean(pd.Series(recall)))
# print("\nMean Cross validation Accuracy score", np.mean(pd.Series(accuracy)))
# #
# # Pickling results
# print("\nPickling stuff...")
# createPickle(accuracy, 'accuracy.pkl')
# createPickle(prec, 'prec.pkl')
# createPickle(results, 'results.pkl')
# createPickle(matrix, 'matrix.pkl')
