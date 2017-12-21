
from __future__ import division

import csv
import numpy as np
import pickle
import time
import os
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TRAINING_LINE_NUMBER = 10 # Number of lines to be read from the huge file, set to total file length while running for entire file
INPUT_FILE_PATH='/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/'
SKIP_FIRST_LINE = True # To skip the first line, as its the header
years = [2003,2004,2005, 2006, 2007, 2008] # Add more years in this list and add the files in the INPUT_FILE_PATH
needed_cols = [0,1, 2, 3, 4, 8, 10, 15, 16, 17]
file_names=[]

timestr = time.strftime("%Y%m%d-%H%M%S")

key_data= np.genfromtxt("/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/Secret_Keys/Key_data.txt", delimiter=",", dtype=str)
conn = S3Connection(key_data[0], key_data[1])


#
#
# try:
#     path = "/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/plane-data.csv"
#     dfPlane = pd.read_csv(path)
#     path = '/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/airports.csv'
#     dfAirport = pd.read_csv(path)
#     path = '/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/carriers.csv'
#     dfCarrier = pd.read_csv(path)
# except Exception as e:
#     print ("Supplemental Data Import failed", e)
#
#
# # Readng the main file in a Pandas dataframe
#
# try:
#     for year in YEARS:
#         path = os.path.join(INPUT_FILE_PATH, '%d.csv' % int(year))
#         dfMaster = pd.read_csv(path, nrows=TRAINING_LINE_NUMBER,skiprows=0,encoding = "ISO-8859-1",usecols=[u'Year', u'Month', u'DayofMonth', u'DayOfWeek',u'UniqueCarrier',u'DepTime',u'TailNum',u'Origin', u'Dest',u'DepDelay'])
# except Exception as e:
#     print ("Supplemental Data Import failed", e)
# dfMaster.head()
#

#
# function: ComputeDayofYear()
# description: This function will return an integer to represent the day of the year given an integer
#    representing month and an integer representing the day of the month.  This number will
#    correspond to the ordered day of the year [0-365].  For instance, Jan 1st will be returned
#    as 0.  Feb 29th will be returned as 59.
# input: row of csv file, a raw dataset
# output: row of csv file, date of year value of which is encoded.
#


def ComputeDayofYear(row):
    if(row[1] == '1'):
        calc = 0 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '2'):
        calc = 31 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '3'):
        calc = 60 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '4'):
        calc = 91 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '5'):
        calc = 121 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '6'):
        calc = 152 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '7'):
        calc = 182 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '8'):
        calc = 213 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '9'):
        calc = 244 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '10'):
        calc = 274 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '11'):
        calc = 305 + int(row[2]) - 1
        row[2] = str(calc)
    elif(row[1] == '12'):
        calc = 335 + int(row[2]) - 1
        row[2] = str(calc)
    return row


#
# function: DiscretizeDepTime()
# description: This function takes a scheduled departure time, classifies the departure time as:
#    morning (0700 - 1259), afternoon (1300 - 1759), or evening (1800-0659).  The input value
#    is assumed to be an integer in 24-hour time format.  These labels will correspond to
#    variable values of 0 = morning, 1 = afternoon, 2 = evening.  The value is then returned.
#    An error time is returned as morning.
# input: row of csv file, a raw dataset
# output: row of csv file, departure time value of which is encoded.
#

def DiscretizeDepTime(row):

    if(int(row[4]) <= 559):
        row[4] = '2'
    elif(int(row[4]) >= 600 and int(row[4]) <= 1259):
        row[4] = '0'
    elif(int(row[4]) >= 1300 and int(row[4]) <= 1759):
        row[4] = '1'
    elif(int(row[4]) >= 1800):
        row[4] = '2'
    else:
        row[4] = '0'
    return row

#
# function: AddDepVar()
# description: This function adds a classification label based on the length of the recorded
#    Departure Delay in the data set.  It assumes an input integer value of the delay in mins.
#    By airline industry standards, flight delays are defined as departure delays greater than
#    or equal to 15 minutes.  For delayed flights, this variable will have value "1".
#    For on time flights, it will have value "0".  Default value will be set at "0".
# input: row of csv file, a raw dataset
# output: row of csv file, delay value of which is encoded as binary.
#


def AddDepVar(row):

    if(row[7] >= '15'):
        row[7] = '1'
    else:
        row[7] = '0'
    return row

#
# function: SaveData()
# description: This function pickles each file. Also, due to the lack of storage space on local server, it stores data to S3 server as well.
# input: data= data structure which will be stored for future uses
#        pickle_file_name= file name to be used to store data
# output: null
#


def createPickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print("Pickled", filename)


def getTailNum(TailNum, inTailNum):
    out = []
    for x, y in inTailNum.iteritems():
        out.append(TailNum.get_value(y))
    return out


def getDest(Dest, inDest):
    out = []
    for x, y in inDest.iteritems():
        out.append(Dest.get_value(y))
    return out


def getOrigin(Origin, inOrign):
    out = []
    for x, y in inOrign.iteritems():
        out.append(Origin.get_value(y))
    return out


def getCarrier(UniqueCarrier, inCarrier):
    out = []
    for x, y in inCarrier.iteritems():
        out.append(UniqueCarrier.get_value(y))
    return out

def convert(findata):
    dfMaster = pd.DataFrame(np.array(findata))

    dfMaster.fillna(0, inplace=True)
    dfMaster[1] = dfMaster[1].astype('int')
    dfMaster[2] = dfMaster[2].astype('int')
    dfMaster[3] = dfMaster[3].astype('int')
    dfMaster[4] = dfMaster[4].astype('int')
    dfMaster[7] = dfMaster[7].astype('int')

    df = dfMaster
    le = LabelEncoder()
    # train[c] = le.fit_transform(train[c])

    # Converting categorical data to numeric for cols - TailNum,
    # UniqueCarrier, Dest, Origin
    print("Converting categorical data to numeric...")
    for col in set(df.columns):
        if df[col].dtype == np.dtype('object'):
            # df[col] = le.fit_transform(df[col])
            #
            # print("Converting...", col)
            if col == 6:
                print("inside5")
                s = np.unique(df[col].values)
                TailNum = pd.Series([x[0] for x in enumerate(s)], index=s)
                createPickle(TailNum, 'TailNum_2008.pkl')
                df[6] = getTailNum(TailNum, df[6])
            if col == 5:
                s = np.unique(df[col].values)
                UniqueCarrier = pd.Series([x[0] for x in enumerate(s)], index=s)
                createPickle(UniqueCarrier, 'UniqueCarrier_2008.pkl')
                df[5] = getCarrier(UniqueCarrier, df[5])
            if col == 9:
                s = np.unique(df[col].values)
                Dest = pd.Series([x[0] for x in enumerate(s)], index=s)
                createPickle(Dest, 'Dest_2008.pkl')
                df[9] = getDest(Dest, df[9])
            if col == 8:
                s = np.unique(df[col].values)
                Origin = pd.Series([x[0] for x in enumerate(s)], index=s)
                createPickle(Origin, 'Origin_2008.pkl')
                df[8] = getOrigin(Origin, df[8])

    findata = df.values
    return findata


def SaveData(data, pickle_file_name):

    f = open(pickle_file_name, "wb")
    try:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)
    f.close()

    bucket = conn.get_bucket('sdmairlines')
    k = Key(bucket)
    k.key = pickle_file_name
    k.set_contents_from_filename(pickle_file_name)

    os.remove(pickle_file_name)

def loadData(fileName):
    print('loading data from AWS to pickle File')
    if os.path.exists(fileName) == False:
        bucket = conn.get_bucket('sdmairlines')
        k = Key(bucket)
        k.key = fileName
        k.get_contents_to_filename(fileName)

    print('now unpickle...')
    x = pickle.load(open(fileName, "rb"))
    x = np.array(x)
    print('x.shape = ', x.shape, x[:, -1:].shape)
    y = x[:, -1:].copy()  # last col is y value (delay or not)
    x[:, -1:] = 1.
    return x, y



data1 = []
data = []
for i in years:
    data1 = []
    data = []
    file_path = os.path.join(INPUT_FILE_PATH, '%d.csv' % int(i))
    pickle_file_name = 'Data-' + str(i)
    file_names.append(pickle_file_name)
    with open(file_path, 'r') as data_csv:
        csv_reader = csv.reader(data_csv, delimiter=',')
        j = 0
        for row in csv_reader:
            # and j<80000000: #and (row[16] == 'SFO' or row[16] == 'OAK'):
            if row[21] == '0':
    #                 if (row[16] == 'SFO' or row[16] == 'OAK'):
                    content = [row[i] for i in needed_cols]
                    content2 = ComputeDayofYear(content)
                    content3 = DiscretizeDepTime(content2)
                    content4 = AddDepVar(content3)
                    data.append(content4)
                    ##temp = convert(data)
                    #data1.append(convert(data).tolist())
                    #data = []
                    # print 'content4', content4
                    # print 'data', data
                    # fff = raw_input()

                    j = j + 1
                    if (j % 1000000 == 0):
                        SaveData(convert(data).tolist(), pickle_file_name + '-' + str(j))
                        print(pickle_file_name+'-'+str(j))
                        data1 = []
                        data = []

                    if(j == 4000000):
                        break
    #SaveData(data, pickle_file_name)
#
# for name in ['Data-2005-500', 'Data-2005-1000','Data-2006-500', 'Data-2006-1000']:
#      X, Y = loadData(name)
#      print(X)
#      print(Y)