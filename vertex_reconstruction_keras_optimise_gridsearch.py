# Vertex reconstruction with keras deep neural network
# Performs optimisation of individual hyperparameters
# To run:
# python vertex_reconstruction_keras_optimise_gridsearch.py mcinfile.csv hitinfile.csv 

# imports
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing
from multiprocessing import Pool

from array import array
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Concatenate, Input, Lambda, Masking, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

tf.debugging.set_log_device_placement(True)

# import the data from the root file
mc_infile = sys.argv[1]
hit_infile = sys.argv[2]
#outfile   = sys.argv[3] #check which file format is output

# set TF random seed to improve reproducibility
seed = 150
np.random.seed(seed)

# define the number of outputs and hit features
num_outputs  = 3 # mcx,mcy,mcz,mcu,mcv,mcw,mct
num_features = 5 # pmtX,pmtY,pmtZ,pmtT,pmtQ

print("opening %s and %s with input variables"%(mc_infile,hit_infile))
hitfile = open(str(hit_infile))
hitdata = pd.read_csv(hitfile,header=None,comment='#',sep='\n')
hitdata = hitdata[0].str.split(',',expand=True).fillna(0)
num_hits = len(hitdata.columns)
train_x = hitdata.values
train_x_vtx = train_x[:,:3,:]
train_x_t = train_x[:,3,:]
train_x_q = train_x[:,4,:]
mcfile = open(str(mc_infile))
train_y = np.array(pd.read_csv(mcfile,header=None,comment='#'))

# scale the input variables since deep learning model uses a weighted sum of input variables
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
train_x = scaler.fit_transform(train_x)
train_y = scaler.fit_transform(train_y)
x_samples = int(len(hitdata)/5.)
train_x = train_x.reshape(x_samples,num_hits,num_features)
print(train_x.shape)
print(train_y.shape)

# Dense class creates fully-connected layers
# ReLU is the rectified linear unit activation function - will output the input directly if positive, or zero otherwise
def create_model(optimizer='Adamax', init_mode='he_uniform',activation='relu',neurons1=50,neurons2=25,loss='mean_squared_error'):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Masking(mask_value=0,input_shape=(num_hits,num_features)))
    model.add(Flatten())
    model.add(Dense(neurons1, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(neurons2, kernel_initializer=init_mode, activation=activation))
    model.add(Dense(num_outputs, kernel_initializer=init_mode, activation=activation)) # final output has 7 dimensions
    # Print model summary
    model.summary()
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

estimator = KerasRegressor(build_fn=create_model, epochs=10, batch_size=72, verbose=0) #epochs: no. of times to iterate over training data arrays; batch_size = no. of samples per gradient update;

# define the grid search parameters: comment out one option at a time to reduce CPU time
# change grid search parameters as applicable for optimisation
# 1. Tune batch_size, epochs
batch_size = [64, 128, 256, 512]
epochs = [8,16, 32, 64,]

# 2. Tune the training optimisation algorithm, the weight initialisation
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#param_grid = dict(optimizer=optimizer, init_mode=init_mode)

# 3. Tune the activation function, the number of neurons for the first layer
#activation = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
neurons1 = [25, 50, 60, 70, 80, 90, 100]
#param_grid = dict(activation=activation, neurons1=neurons1)

# 4. Tune the number of neurons for the 2nd layer
neurons2 = [5, 10, 15, 20, 25, 30]
#param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, init_mode=init_mode, activation=activation, neurons1=neurons1, neurons2=neurons2)

# 5. Tune the loss function
loss = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity']

param_grid = dict(loss=loss)

# search the grid parameters:
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(train_x,train_y)
#summarise results
print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


##checkpoint
#filepath = "file"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
#callbacks_list = [checkpoint]
## Fit the model
#model.fit(train_x, train_y, validation_split=0.33, epochs=50, batch_size=2, callbacks=callbacks_list, verbose=0)

# ------------------------------------------
