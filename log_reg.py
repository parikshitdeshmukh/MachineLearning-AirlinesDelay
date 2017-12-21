# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 03:18:50 2017

@author: anu21
"""

#import pickle
#from boto.s3.connection import S3Connection
#from boto.s3.key import Key
#conn = S3Connection('...', '..')
#bucket = conn.get_bucket('sdmairlines')
#k = Key(bucket)
#k.key = 'Data-2005'
#k.get_contents_to_filename('Data-2005')
#theta = pickle.load(open('Data-2005', 'rb'))


import numpy as np
import pandas as pd
import random
import pickle
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import csv


key_data= np.genfromtxt("/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/Secret_Keys/Key_data.txt", delimiter=",", dtype=str)
pic = ['Data-2003-1000000']
pictest = ['Data-2006-1000000','Data-2006-2000000']
conn = S3Connection(key_data[0], key_data[1])
bucket = conn.get_bucket('sdmairlines')
k = Key(bucket)
theta = []

def gradientDescent(x, y, numIterations, dimension, theta):
    for i in range(1, numIterations):
        randIdx = random.randint(0, len(x) - 1)
        xTrans = x[randIdx][np.newaxis].transpose()
        u = 1 / (1 + np.exp(np.dot(theta.transpose().astype(float) * (-1), xTrans.astype(float))))
        loss = y[randIdx] - u
        gradient = np.dot(loss[0][0], xTrans)
        # update
        theta = theta + gradient / i
    return theta

def loadData(x):
    x = np.array(x)
    print('x.shape = ', x.shape, x[:, -1:].shape)
    y = x[:, 7].copy()
    x[:,7] = 1
    x[:,0] = 1

    return x, y

def typeconvert(findata):
    dfMaster = pd.DataFrame(np.array(findata))

    dfMaster.fillna(0, inplace=True)
    dfMaster[0] = 1
    dfMaster[1] = dfMaster[1].astype('int')
    dfMaster[2] = dfMaster[2].astype('int')
    dfMaster[3] = dfMaster[3].astype('int')
    dfMaster[4] = dfMaster[4].astype('int')
    dfMaster[5] = dfMaster[5].astype('int')
    dfMaster[6] = dfMaster[6].astype('int')
    dfMaster[7] = dfMaster[7].astype('int')
    dfMaster[8] = dfMaster[8].astype('int')
    dfMaster[9] = dfMaster[9].astype('int')

    findata = dfMaster.values
    return findata


##Training the model
for el in pic:
    k.key = el
    k.get_contents_to_filename(el)
    findata = pickle.load(open(el, 'rb'))
    findata = typeconvert(findata)
    #cols = list(df.loc[:,'0':'6']) + list(df.loc[:,'7':'9'])
    x,y = loadData(findata)
    #x = df[cols]#.columns[0,1,2,3,4,5,7,8]
    #y = df[6]
    print('x.shape = ', x.shape, 'y.shape = ', y.shape)
    if len(theta)==0:
        theta = np.zeros(x.shape[1])[np.newaxis].transpose()
        print('theta == None...... initialize..........', theta.shape)
    theta = gradientDescent(x, y, 500000, x.shape[1], theta)
    print('finished gradientDescent of ', el)
print('theta', theta)



##Testing the model (predict with test dataset)
accu = 0.
length = 0.
tp, tn, fp, fn = 0., 0., 0., 0.
for elem in pictest:
    k.key = elem
    k.get_contents_to_filename(elem)
    findata = pickle.load(open(elem, 'rb'))
    findata = typeconvert(findata)
    #findata = convert(findata)
    x,y = loadData(findata)
    dotProduct = np.dot(x, theta)
    print('============= dot product =============')
    print(dotProduct)
    print('=============y =============')
    print(y)
    
    reverseLogit = [np.exp(dot) / (1 + np.exp(dot)) for dot in dotProduct]
    prob = [1 if rev >= 0.5 else 0 for rev in reverseLogit]
    for i in range(len(prob)):
        if prob[i] == 1 and y[i] == 1:
            accu += 1
            tp += 1
        elif prob[i] == 1 and y[i] == 0:
            fp += 1
        elif prob[i] == 0 and y[i] == 1:
            fn += 1
        elif prob[i] == 0 and y[i] == 0:
            accu += 1
            tn += 1
        else:
            raise Exception('wtf!!!', prob[i], y[i])
    length += len(prob)
# print accuracy, precision, and recall
print('accuracy = ', accu * 100 / length, (tp + tn) / (tp + fp + fn + tn))
print('precision = ', tp / (tp + fp))
print('recall = ', tp / (tp + fn))

specf = tn / (tn + fp)
sens = tp / (tp + fn)
print('Specificity = ', specf)
print('Sensitivity = ', sens)

## plot ROR which is sensitivity(y) vs specifictity

p = tp / (tp + fp)
r = tp / (tp + fn)
print('precision = ', p)
print('recall = ', r)
print('F-Score= ', 2*((p*r)/(p+r)))