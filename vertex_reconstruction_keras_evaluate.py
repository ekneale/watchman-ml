# Evaluate vertex reconstruction
# To run:
# python vertex_reconstruction_keras_evaluate.py predictioninfile.csv mcinfile.csv

# imports
from __future__ import print_function
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt, pow

from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Lambda, Masking, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Define the number of outputs
num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct

# Get the filenames
prediction_infile = sys.argv[1]

print("opening %s with truth and prediction variables"%(prediction_infile))

# Read in data
infile = open(str(prediction_infile))
data = pd.read_csv(infile)
print(data)
index,mcx,mcy,mcz,x,y,z = np.hsplit(np.array(data),7)

# Score with sklearn
x_score_sklearn = np.sqrt(metrics.mean_squared_error(x,mcx))
print('RMSE (sklearn): {0:f}'.format(x_score_sklearn))
y_score_sklearn = np.sqrt(metrics.mean_squared_error(y,mcy))
print('RMSE (sklearn): {0:f}'.format(y_score_sklearn))
z_score_sklearn = np.sqrt(metrics.mean_squared_error(z,mcz))
print('RMSE (sklearn): {0:f}'.format(z_score_sklearn))

def deltax(data):
    return data[1] - data[4]
def deltay(data):
    return data[2] - data[5]
def deltaz(data):
    return data[3] - data[6]
def deltaxyz(data):
    return np.sqrt(pow(data[1]-data[4],2)+pow(data[2]-data[5],2)+pow(data[3]-data[6],2))

dx = data.apply(deltax,axis=1)
dy = data.apply(deltay,axis=1)
dz = data.apply(deltaz,axis=1)
dxyz = data.apply(deltaxyz,axis=1)

dxplt = plt.figure(1)
plt.hist(dx,500)
dxplt.show()

dyplt = plt.figure(2)
plt.hist(dy,500)
dyplt.show()

dzplt = plt.figure(3)
plt.hist(dz,500)
dzplt.show()

dxyzplt = plt.figure(4)
plt.hist(dxyz,500)
dxyzplt.show()

#keep figures alive
input()
