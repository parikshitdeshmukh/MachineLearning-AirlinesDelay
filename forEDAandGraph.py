import numpy as np
import pandas as pd
import random
import pickle
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import csv


INPUT_FILE_PATH = "/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/airline/"  # Unix pathSKIP_FIRST_LINE = True  # To skip the first line, as its the header

key_data= np.genfromtxt("/media/parik/New Volume/SDM/R Lab/Final Project Ariline Delay/Data-Mining-Project-master/Secret_Keys/Key_data.txt", delimiter=",", dtype=str)
pic = ['Data-2003-1000000', 'Data-2003-2000000', 'Data-2003-3000000','Data-2003-4000000']
##pictest = ['Data-2006-1000000','Data-2006-2000000']
conn = S3Connection(key_data[0], key_data[1])
bucket = conn.get_bucket('sdmairlines')
k = Key(bucket)
theta = []

master = []

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

    #
    #
    # dfMaster = pd.DataFrame(np.array(findata), columns=['Year', 'Month', 'DayOfYear', 'DayOfWeek', 'DepTime', 'UniqueCareer', 'TailNumber','DepDelay', 'Origin', 'Dest'])
    # dfMaster.fillna(0, inplace=True)
    #
    # dfMaster['Year'] = dfMaster['Year'].astype('int')
    # dfMaster['Month'] = dfMaster['Month'].astype('int')
    # dfMaster['DayOfYear'] = dfMaster['DayOfYear'].astype('int')
    # dfMaster['DayOfWeek'] = dfMaster['DayOfWeek'].astype('int')
    # dfMaster['DepTime'] = dfMaster['DepTime'].astype('int')
    # dfMaster['UniqueCareer'] = dfMaster['UniqueCareer'].astype('int')
    # dfMaster['TailNumber'] = dfMaster['TailNumber'].astype('int')
    # dfMaster['DepDelay'] = dfMaster['DepDelay'].astype('int')
    # dfMaster['Origin'] = dfMaster['Origin'].astype('int')
    # dfMaster['Dest'] = dfMaster['Dest'].astype('int')
    #
    #
    # findata = dfMaster
    # return findata

try:
    for el in pic:
        k.key = el
        k.get_contents_to_filename(el)
        findata = pickle.load(open(el, 'rb'))
        findata = typeconvert(findata)
        master.append(findata)
except Exception as e:
    print("data appending failed", e)



df = pd.concat(master, ignore_index= True)
rows = np.random.choice(np.random.permutation(df.index.values), len(df) // 2,replace=False)  # 33% sampling of training data
sampleDF = df.ix[rows]
sampleDF.columns = ['Year', 'Month', 'DayOfYear', 'DayOfWeek', 'DepTime', 'UniqueCareer', 'TailNumber','DepDelay', 'Origin', 'Dest']

csvFile = sampleDF
csvFile.to_csv(INPUT_FILE_PATH + "NewDATA" + ".csv", index=False)

################## Graphs,....


