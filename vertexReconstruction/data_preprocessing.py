from model_params import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def prep_datasets(X, Y, x_samples, num_hits):
    
    # Reshape the data into hit information for each event
    N = max_hits-num_hits
    # pad training data to have num_hits = max_hits (currently hard-coded)
    X = np.pad(X, [(0,0),(0,N)], mode='constant',constant_values='nan')
    X = X.reshape(x_samples,num_features,max_hits)
    X = X.astype('float32')
    Y = Y.astype('float32')

    # group into xyz, t and q for scaling
    # xyz should be split if detector is not of equal dimensions
    X_vtx = X[:,:3,:]
    X_t = X[:,3,:]
    X_q = X[:,4,:]

    '''
    # Plot the distribution (e.g. z) prior to scaling
    plt.hist(X_vtx[:,2][~np.isnan(X_vtx[:,2])],bins=500,histtype='step')
    plt.title('True z at hit PMT',fontsize=20)
    plt.xlabel('True z (cm)',fontsize=18)
    plt.savefig('Z.png')
    plt.show()
    '''

    # Scale the training input and output variables
    x_scaler_vtx = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_t = preprocessing.MinMaxScaler(feature_range=(-1,1))
    x_scaler_q = preprocessing.MinMaxScaler()
    y_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

    X_vtx = x_scaler_vtx.fit_transform(X_vtx.reshape(-1,X_vtx.shape[-1])).reshape(X_vtx.shape)
    X_t = x_scaler_t.fit_transform(X_t)
    X_q = x_scaler_q.fit_transform(X_q)
    Y = y_scaler.fit_transform(Y)


    '''
    #Plot the distribution (e.g. z) post-scaling
    plt.hist(X_vtx[:,2][~np.isnan(X_vtx[:,2])],bins=500,histtype='step')
    plt.title('Scaled z at hit PMT',fontsize=20)
    plt.yscale('log')
    plt.xlabel('Scaled z',fontsize=18)
    plt.savefig('z_scaled.png')
    plt.show()
    '''

    # Restore the shape of the training data post scaling
    X = np.hstack((X_vtx[:,0],X_vtx[:,1],X_vtx[:,2],X_t,X_q))
    X = X.reshape(x_samples,num_features,max_hits)

    # Replace all 'nan' values with 0 ready for masking layer
    X[np.isnan(X)]=0.

    return X,Y


