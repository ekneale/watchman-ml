# Vertex reconstruction with keras deep neural network.
# Performs simultaneous optimisation of hyperparameters
# which can be subsequently input in the training and 
# prediction code.
# Author Elisabeth Kneale, November 2020
# To run:
# python vertex_reconstruction_keras_optimise_hyperas.py mcinfile.csv hitinfile.csv 

# imports
from __future__ import print_function
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout,Input,Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, quniform


#tf.debugging.set_log_device_placement(True)

# set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)


def data():
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
    bs_infile = sys.argv[3]
    
    bsfile = open(str(bs_infile))
    bs_X = np.array(pd.read_csv(bsfile,header=None,comment='#'))


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
    return x_train, y_train, x_test, y_test

# Dense class creates fully-connected layers
# ReLU is the rectified linear unit activation function - will output the input directly if positive, or zero otherwise
def model(x_train,y_train,x_test,y_test):
    """
    Model providing function.
    Wrap the parameters you want to optimize into double curly brackets and choose 
    a distribution over which to run the algorithm.
    """
    keras.backend.clear_session()
    # define the number of outputs and hit features
    num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
    num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ
    
    hits=Input(shape=(num_features,num_hits))
    mask=Masking(mask_value=0.)(hits)
    flat1 = Flatten()(mask)

    bonsai_vtx = Input(shape=(3))

    hidden1 = Dense(int({{quniform(420,512,1)}}), kernel_initializer={{choice(['normal', 'he_normal','uniform','he_uniform'])}}, activation={{choice(['relu','linear'])}})(flat1)
    hidden2 = Dense(int({{quniform(320,420,1)}}), kernel_initializer={{choice(['normal', 'he_normal','uniform','he_uniform'])}}, activation={{choice(['relu','linear'])}})(hidden1)
    hidden3 = Dense(int({{quniform(1,128,1)}}), kernel_initializer={{choice(['normal', 'he_normal','uniform','he_uniform'])}}, activation={{choice(['relu','linear'])}})(hidden2)
    merge = Concatenate()([hidden2,bonsai_vtx])
    output = Dense(num_outputs, kernel_initializer={{choice(['normal', 'he_normal','uniform','he_uniform'])}},activation={{choice(['relu','linear'])}})(merge) # final output has 7 dimensions
    # Print model summary
    model = Model(inputs=[hits,bonsai_vtx],outputs=output)
    # Compile model
    # mean_squared_error for metrics (potentially more informative than accuracy)
    model.compile(loss='mean_squared_error', optimizer= 'Adamax', metrics=['accuracy'])

    print(model.metrics_names)
#    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/fit', histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_loss',patience=10)
    checkpointer = ModelCheckpoint(filepath='keras_weights_optimisation.hdf5',
            verbose=1,
            save_best_only=True)

    result = model.fit(x_train,y_train, 
            batch_size=int({{quniform(2,256,1)}}),
            epochs=int({{quniform(2,256,1)}}),
            verbose=2,
            validation_data= (x_test,y_test),
            callbacks=[early_stopping,checkpointer])

    mse,acc = model.evaluate(x_test,y_test,verbose=0)

    print('mse,acc:',mse,acc)
    return {'loss':mse, 'status':STATUS_OK, 'model':model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=500,
                                          trials=Trials(),
                                          keep_temp=True)
    x_train, y_train, x_test, y_test = data()

    with open("optimisation_singles.txt","w") as f:
        print("Evaluation of best performing model:",file=f)
        print(best_model.evaluate(x_test, y_test),file=f)
        print("Best performing model chosen hyper-parameters:",file=f)
        print(best_run,file=f)


