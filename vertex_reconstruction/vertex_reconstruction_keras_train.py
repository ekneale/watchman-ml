# Vertex reconstruction with keras deep neural network
# To run:
# python vertex_reconstruction_keras_train.py mcinfile.csv hitinfile.csv 

# imports
from __future__ import print_function
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Lambda, Masking, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

#tf.debugging.set_log_device_placement(True)

# set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)
#tf.set_random_seed(seed)


# define the number of outputs and hit features
num_outputs  = 3 # mcx,mcy,mcz (mcu,mcv,mcw,mct)
num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ

# import the data from the root file
mc_infile = sys.argv[1]
hit_infile = sys.argv[2]
#outfile   = sys.argv[3] #check which file format is output


print('opening %s and %s with input variables'%(mc_infile,hit_infile))
# read in trainng data with the same number of columns per line 
# (fill empty columns with nan so it will be ignored by scaler)
hitfile = open(str(hit_infile))
hitdata = pd.read_csv(hitfile,header=None,comment='#',sep='\n')
hitdata = hitdata[0].str.split(',',expand=True).fillna('nan')
X = hitdata.values
mcfile = open(str(mc_infile))
Y_tmp = np.array(pd.read_csv(mcfile,header=None,comment='#'))
Y = Y_tmp[:,:3] # we only want mcx, mcy and mcz (first three columns) for now

# Reshape the data into hit information for each event
x_samples = int(len(hitdata)/5.)
num_hits = len(hitdata.columns)
X = X.reshape(x_samples,num_features,num_hits)
X = X.astype('float32')
Y = Y.astype('float32')

# group into xyz, t and q for scaling
print(X)
x_train_vtx = X[:,:3,:]
x_train_t = X[:,3,:]
x_train_q = X[:,4,:]

# Plot the distribution (e.g. q) prior to scaling
plt.hist(x_train_q[~np.isnan(x_train_q)],range=(0,5),bins=50,histtype='step')
plt.title('True charge at hit PMT',fontsize=20)
plt.xlabel('True Q (pe)',fontsize=18)
plt.savefig('Q.png')
plt.show()

# Scale the training input and output variables 
x_scaler_vtx = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_t = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_q = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
#y_scaler = preprocessing.StandardScaler()

x_train_vtx = x_scaler_vtx.fit_transform(x_train_vtx.reshape(-1,x_train_vtx.shape[-1])).reshape(x_train_vtx.shape)
x_train_t = x_scaler_t.fit_transform(x_train_t)#.reshape(-1,x_train_t.shape[-1])).reshape(x_train_t.shape)
x_train_q = x_scaler_q.fit_transform(x_train_q)#.reshape(-1,x_train_q.shape[-1])).reshape(x_train_q.shape)
y_train = y_scaler.fit_transform(Y)

#Plot the distribution post-scaling  
plt.hist(x_train_q[~np.isnan(x_train_q)],range=(0,0.25),bins=50,histtype='step')
plt.title('Scaled Q at hit PMT',fontsize=20)
plt.xlabel('Scaled Q',fontsize=18)
plt.savefig('Q_scaled.png')
plt.show()

# Get the original values and reshape for verification
x_train_vtx_test = x_scaler_vtx.inverse_transform(x_train_vtx.reshape(-1,x_train_vtx.shape[-1])).reshape(x_train_vtx.shape)
x_train_t_test = x_scaler_t.inverse_transform(x_train_t)
x_train_q_test = x_scaler_q.inverse_transform(x_train_q)
x_train_test = np.hstack((x_train_vtx_test[:,0,:],x_train_vtx_test[:,1,:],x_train_vtx_test[:,2,:],x_train_t_test,x_train_q_test))
x_train_test = x_train_test.reshape(x_samples,num_features,num_hits)
print(x_train_test)

# Restore the shape of the training data post scaling
x_train = np.hstack((x_train_vtx[:,0,:],x_train_vtx[:,1,:],x_train_vtx[:,2,:],x_train_t,x_train_q))
x_train = x_train.reshape(x_samples,num_features,num_hits)

# Replace all 'nan' values with 0 ready for masking layer
x_train[np.isnan(x_train)]=0.

def model():
    '''
    Model providing function.
    Wrap the parameters you want to optimize into double curly brackets and choose 
    a distribution over which to run the algorithm.
    '''
    keras.backend.clear_session() # clear the previous model to free up GPU space
    # Dense class creates fully-connected layers
    # Define the number of outputs and hit features
    num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
    num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ

    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(num_features,num_hits)))
    model.add(Flatten())
    model.add(Dense(104, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(57, kernel_initializer= 'uniform', activation='relu'))
    model.add(Dense(num_outputs, kernel_initializer='he_normal', activation='relu')) # final output has 3 dimensions
    # Print model summary
    model.summary()
    # Compile model
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity'
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])


    return model 


estimator=KerasRegressor(build_fn=model,
              batch_size=113,
              epochs=57,
              verbose=0)

kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(estimator, x_train, y_train, cv = kfold)
print('Results: %.2f (%.2f) MSE' % (results.mean(), results.std()))

# Fit the model
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/train', histogram_freq=1)
filepath = 'weights_first_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only = True, save_weights_only = True, mode='auto')
callbacks_list = [checkpoint, tbCallBack]
history = estimator.fit(x_train,y_train,
            batch_size=113,
            epochs=57,
            validation_split=0.33,
            callbacks=callbacks_list,
            verbose=2)

