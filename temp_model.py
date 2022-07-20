#coding=utf-8

from __future__ import print_function

try:
    import sys
except:
    pass

try:
    import glob
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import pandas as pd
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    from array import array
except:
    pass

try:
    from sklearn import datasets
except:
    pass

try:
    from sklearn import metrics
except:
    pass

try:
    from sklearn import model_selection
except:
    pass

try:
    from sklearn import preprocessing
except:
    pass

try:
    from sklearn.model_selection import train_test_split
except:
    pass

try:
    from tensorflow import keras
except:
    pass

try:
    from tensorflow.keras.models import Sequential, Model
except:
    pass

try:
    from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout
except:
    pass

try:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
except:
    pass

try:
    from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, quniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

"""
Separated from create_model() so that hyperopt won't
reload data for each evaluation run.
"""
# define the number of outputs and hit features
num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ

# import the data from the root file
mc_infile = sys.argv[1]
hit_infile = sys.argv[2]
#outfile   = sys.argv[3] #check which file format is output


print("opening %s and %s with input variables"%(mc_infile,hit_infile))
hitfile = open(str(hit_infile))
hitdata = pd.read_csv(hitfile,header=None,comment='#',sep='\n')
# read in with the same number of columns per line (fill empty columns with zeros)
hitdata = hitdata[0].str.split(',',expand=True).fillna('nan')
X = hitdata.values
mcfile = open(str(mc_infile))
Y_tmp = np.array(pd.read_csv(mcfile,header=None,comment='#'))
Y = Y_tmp[:,:3]

# Reshape the data into hit information for each event
x_samples = int(len(hitdata)/5.)
num_hits = len(hitdata.columns)
X = X.reshape(x_samples,num_features,num_hits)
X = X.astype('float32')
Y = Y.astype('float32')
# Group into xyz, t and q for scaling
# xyz should be split if detector not of equal dimensions
X_vtx = X[:,:3,:]
X_t = X[:,3,:]
X_q = X[:,4,:]

# scale the input variables since deep learning model uses a weighted sum of input variables
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
x_scaler_vtx = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_t = preprocessing.MinMaxScaler(feature_range=(-1,1))
x_scaler_q = preprocessing.MinMaxScaler()
y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

X_vtx = x_scaler_vtx.fit_transform(X_vtx.reshape(-1,X_vtx.shape[-1])).reshape(X_vtx.shape)
X_t = x_scaler_t.fit_transform(X_t)
X_q = x_scaler_q.fit_transform(X_q)

Y = y_scaler.fit_transform(Y)

# Restore the shape of the data post scaling
X = np.hstack((X_vtx[:,0,:],X_vtx[:,1,:],X_vtx[:,2,:],X_t,X_q))
X = X.reshape(x_samples,num_features,num_hits)
# Replace all 'nan' values with 0 ready for masking layer
X[np.isnan(X)]=0
x_train,x_test = train_test_split(X,random_state=1)
y_train,y_test = train_test_split(Y,random_state=1)


def keras_fmin_fnct(space):

    """
    Model providing function.
    Wrap the parameters you want to optimize into double curly brackets and choose 
    a distribution over which to run the algorithm.
    """
    keras.backend.clear_session()
    # define the number of outputs and hit features
    num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
    num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ
    
    model = Sequential()
    model.add(Masking(mask_value=0,input_shape=(num_features,num_hits)))
    model.add(Flatten())
    model.add(Dense(int(space['int']), kernel_initializer=space['kernel_initializer'], activation=space['activation']))
    model.add(Dense(int(space['int_1']), kernel_initializer=space['kernel_initializer_1'], activation=space['activation_1']))
    model.add(Dense(int(space['int_2']), kernel_initializer=space['kernel_initializer_2'], activation=space['activation_2']))
    model.add(Dense(num_outputs, kernel_initializer=space['kernel_initializer_3'],activation=space['activation_3'])) # final output has 7 dimensions
    # Print model summary
    model.summary()
    # Compile model
    # mean_squared_error for metrics (potentially more informative than accuracy)
    model.compile(loss='mean_squared_error', optimizer= 'Adamax', metrics=['accuracy'])
    print(model.metrics_names)
#    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/fit', histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss',patience=20)
    checkpointer = ModelCheckpoint(filepath='keras_weights_optimisation.hdf5',
            verbose=1,
            save_best_only=True)

    result = model.fit(x_train,y_train, 
            batch_size=int(space['int_3']),
            epochs=int(space['int_4']),
            verbose=2,
            validation_data= (x_test,y_test),
            callbacks=[early_stopping,checkpointer])

    mse,acc = model.evaluate(x_test,y_test,verbose=0)

    print('mse,acc:',mse,acc)
    return {'loss':mse, 'status':STATUS_OK, 'model':model}

def get_space():
    return {
        'int': hp.quniform('int', 420,512,1),
        'kernel_initializer': hp.choice('kernel_initializer', ['normal', 'he_normal','uniform','he_uniform']),
        'activation': hp.choice('activation', ['relu','linear']),
        'int_1': hp.quniform('int_1', 320,420,1),
        'kernel_initializer_1': hp.choice('kernel_initializer_1', ['normal', 'he_normal','uniform','he_uniform']),
        'activation_1': hp.choice('activation_1', ['relu','linear']),
        'int_2': hp.quniform('int_2', 1,128,1),
        'kernel_initializer_2': hp.choice('kernel_initializer_2', ['normal', 'he_normal','uniform','he_uniform']),
        'activation_2': hp.choice('activation_2', ['relu','linear']),
        'kernel_initializer_3': hp.choice('kernel_initializer_3', ['normal', 'he_normal','uniform','he_uniform']),
        'activation_3': hp.choice('activation_3', ['relu','linear']),
        'int_3': hp.quniform('int_3', 2,256,1),
        'int_4': hp.quniform('int_4', 2,256,1),
    }
