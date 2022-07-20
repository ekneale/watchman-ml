# Vertex reconstruction with keras deep neural network
# Creates and stores a model from training data
# Author Elisabeth Kneale, November 2020
# To run:
# python vertex_reconstruction_keras_train.py mcinfile.csv hitinfile.csv 

# imports
from __future__ import print_function
from data_preprocessing import prep_datasets
from model import model
from model_params import *

import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from array import array
#from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score
#from sklearn.metrics import accuracy_score

from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#tf.debugging.set_log_device_placement(True)

# set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)
#tf.set_random_seed(seed)

# Import the training data from the csv files (get_features output)
mc_infile = sys.argv[1]
hit_infile = sys.argv[2]
bs_infile = sys.argv[3]
mc_infile_neutrons = sys.argv[4]

print('opening %s and %s with input and target variables'%(mc_infile,hit_infile))

# First get the input data with the same number of columns per line 
# (number of columns set by max_hits in data_preprocessing.py)
# and fill empty columns with nan so it will be ignored by scaler.
hitfile = open(str(hit_infile))
hitdata = pd.read_csv(hitfile,header=None,comment='#',sep='\n')
hitdata = hitdata[0].str.split(',',expand=True).fillna('nan')
X = hitdata.values
x_samples = int(len(hitdata)/5.)
num_hits = len(hitdata.columns)

bsfile = open(str(bs_infile))
bs_X = np.array(pd.read_csv(bsfile,header=None,comment='#'))


# Then get the target data
mcfile = open(str(mc_infile))
y_tmp = np.array(pd.read_csv(mcfile,header=None,comment='#'))
Y_pos = y_tmp[:,:3] # we only want mcx, mcy and mcz (first three columns) for now
mcfile2 = open(str(mc_infile_neutrons))
y_tmp = np.array(pd.read_csv(mcfile2,header=None,comment='#'))
Y_neut =  y_tmp[:,:3]
Y_tmp = np.hstack((Y_pos,Y_neut))
Y = Y_tmp

# Preprocess the data
x_train,bs_x_train,y_train = prep_datasets(X,bs_X,Y,x_samples,num_hits)

# Build the model
estimator=KerasRegressor(build_fn=model,
              batch_size=batchsize,
              epochs=num_epochs,
              verbose=0)

# Perform k-fold cross validation for less biased estimate of model performance
#kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
#results = cross_val_score(estimator, x_train, y_train, cv = kfold)
#print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))

# Add callbacks
# tensorboard for loss/accuracy plots
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/train', histogram_freq=1)
# early stopping to avoid overtraining
earlystopping = EarlyStopping(monitor='val_loss',patience=20)
# checkpoint to save weights for best model
filepath = 'weights_no_noise_model_first.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only = True, save_weights_only = True, mode='auto')
callbacks_list = [checkpoint, tbCallBack, earlystopping]

# Fit the model
history = estimator.fit([x_train,bs_x_train],y_train,
            batch_size=batchsize,
            epochs=num_epochs,
            validation_split=0.33,
            callbacks=callbacks_list,
            verbose=2)

# Evaluate the model
#results = model.evaluate(x_test,y_test,batch_size=batchsize)
#print("testloss, test acc: ", results)
