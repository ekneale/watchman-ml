# Vertex reconstruction with keras deep neural network
# This reads in the model output by vertex_reconstruction_keras_train.py
# and outputs a prediction for the x, y and z values of the interaction vertex.
# To run:
# python vertex_reconstruction_keras_predict.py mcinfile.csv hitinfile.csv mcinfile_train.csv hitinfile_train.csv  

# imports
from __future__ import print_function
import sys
import glob
import ROOT
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib as plt

from root_numpy import array2root
from root_numpy import fill_hist

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

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, quniform

#tf.debugging.set_log_device_placement(True)

# set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

# Define the number of outputs and hit features
num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ

# Get the filenames
mc_infile = sys.argv[1]
hit_infile = sys.argv[2]
mc_infile_train = sys.argv[3]
hit_infile_train = sys.argv[4]

# Read in training data with the same number of columns per line 
# (fill empty columns with nan)
hitfile_train = open(hit_infile_train)
hitdata_train = pd.read_csv(hitfile_train,header=None,comment='#',sep='\n')
hitdata_train = hitdata_train[0].str.split(',',expand=True).fillna('nan')
x_train = hitdata_train.values
mcfile_train = open(str(mc_infile_train))
y_train_tmp = np.array(pd.read_csv(mcfile_train,header=None,comment='#'))
y_train = y_train_tmp[:,:3] # we only want mcx, mcy and mcz (first three columns) for now

# Read in prediction data with the same number of columns per line (fill empty columns with nan so it will be ignored by scaler)
print("opening %s and %s with truth and input variables"%(mc_infile,hit_infile))
hitfile = open(str(hit_infile))
hitdata = pd.read_csv(hitfile,header=None,comment='#',sep='\n')
hitdata = hitdata[0].str.split(',',expand=True).fillna('nan')
X = hitdata.values


# Add padding to make predict data same size as train data
num_hits = len(hitdata.columns)
num_hits_train = len(hitdata_train.columns)
N = num_hits_train - num_hits
X = np.pad(X, [(0,0),(0,N)], mode='constant',constant_values='nan')

# Read in prediction truth data - TODO save to root file/ntuple
mcfile = open(str(mc_infile))
Y_tmp = np.array(pd.read_csv(mcfile,header=None,comment='#'))
Y = Y_tmp[:,:3] # we only want mcx, mcy and mcz (first three columns) for now

# Reshape data into hit information for each event
x_samples = int(len(hitdata)/5.)
X = X.reshape(x_samples,num_features,num_hits_train)
x_train_samples = int(len(hitdata_train)/5.)
x_train = x_train.reshape(x_train_samples,num_features,num_hits_train)

# group training data into xyz, t and q for scaling
x_train_vtx = x_train[:,:3,:]
x_train_t = x_train[:,3,:]
x_train_q = x_train[:,4,:]

# group prediction data into xyz, t and q for scaling
X_vtx = X[:,:3,:]
X_t = X[:,3,:]
X_q = X[:,4,:]

# Scale the training input and output variables and save the parameters
x_scaler_vtx = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_t = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_q = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
#x_train_vtx = x_train_vtx.astype('float32')
#y_train = y_train.astype('float32')
x_train_vtx = x_scaler_vtx.fit_transform(x_train_vtx.reshape(-1,x_train_vtx.shape[-1])).reshape(x_train_vtx.shape)
x_train_t = x_scaler_t.fit_transform(x_train_t)
x_train_q = x_scaler_q.fit_transform(x_train_q)
y_train = y_scaler.fit_transform(y_train)
# Apply same scaling to the prediction input and output variables
#X = X.astype('float32')
#Y = Y.astype('float32')
X_vtx = x_scaler_vtx.transform(X_vtx.reshape(-1,X_vtx.shape[-1])).reshape(X_vtx.shape)
X_t = x_scaler_t.transform(X_t)
X_q = x_scaler_q.transform(X_q)
Y = y_scaler.transform(Y)

x_predict = np.hstack((X_vtx[:,0,:],X_vtx[:,1,:],X_vtx[:,2,:],X_t,X_q))
x_predict = x_predict.reshape(x_samples,num_features,num_hits_train)
y_predict = Y

# replace all 'nan' values with 0 ready for masking layer
x_predict[np.isnan(x_predict)]=0

keras.backend.clear_session()

# Create the model and load weights from file
# Dense class creates fully-connected layers
# final output has 7 dimensions
model = Sequential()
model.add(Masking(mask_value=0.,input_shape=(num_hits_train,num_features)))
model.add(Flatten())
model.add(Dense(104, kernel_initializer='uniform', activation='softsign'))
model.add(Dense(57, kernel_initializer= 'uniform', activation='softplus'))
model.add(Dense(num_outputs, kernel_initializer='zero', activation='softsign')) 
# Print model summary
model.summary()

# Get weights from trained model
model.load_weights("weights_first_model.hdf5")

# Compile model
model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])

# Make the prediction
y_predicted_scaled = model.predict(x_predict)
y_predicted = y_scaler.inverse_transform(y_predicted_scaled)

# Score with sklearn
score_sklearn = metrics.mean_squared_error(y_predicted, y_predict)
print('MSE (sklearn): {0:f}'.format(score_sklearn))

print(" saving .csv file with vertex variables..")
print("shapes: ", y_train.shape, ",", y_predict.shape, ", ", y_predicted.shape)

prediction=pd.DataFrame(y_predicted)
prediction.to_csv("xyz_uvw_t_Reco.csv", float_format = '%.3f')
print(y_predicted)
x,y,z = y_predicted.T

np.ascontiguousarray(x,'float32')
np.ascontiguousarray(y,'float32')
np.ascontiguousarray(z,'float32')

'''
Read in the tree from the root file made with get_features,
add new branches with the reconstructed x, y and z and save 
to a new root file (so reconstruction can be repeated if needed).
'''
f_old = ROOT.TFile("mcdata_pair_predict.root","read")
tree_old = f_old.Get("data")
f_new = ROOT.TFile("mcdata_pair_predict_update.root","recreate")
tree = tree_old.CloneTree()
n = len(x)

def filltree(tree,x,y,z,n):
    recoX = np.array([x[0]])
    recoY = np.array([y[0]])
    recoZ = np.array([z[0]])
    recox_branch = tree.Branch('recoX',recoX,'recoX/F')
    recoy_branch = tree.Branch('recoY',recoY,'recoY/F')
    recoz_branch = tree.Branch('recoZ',recoZ,'recoZ/F')
    for entry in range(n):
        recoX[0] = x[entry]
        recoY[0] = y[entry]
        recoZ[0] = z[entry]
        tree.Fill()

filltree(tree,x,y,z,n)
tree.Write
f_new.Write()
f_new.Close()
f_old.Close()
