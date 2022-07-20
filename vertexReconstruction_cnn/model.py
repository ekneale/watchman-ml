from model_params import *

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout, Conv2D


def model():
    '''
    Model building function.
    '''
    # Clear the previous model to free up GPU space 
    keras.backend.clear_session()

    # Dense class creates fully-connected layers
    model = Sequential()
    model.add(Masking(mask_value=0.,input_shape=(num_features,max_hits)))
    model.add(Conv2D((num_features,max_hits), kernel_size=5, kernel_initializer=initialiser1, activation=activation1))
    model.add(Flatten())
    model.add(Dense(neurons1, kernel_initializer=initialiser1, activation=activation1))
#    model.add(Dropout(dropout1))
    model.add(Dense(neurons2, kernel_initializer=initialiser2, activation=activation2))
#    model.add(Dropout(dropout2))
    model.add(Dense(neurons3, kernel_initializer=initialiser3, activation=activation3))
#    model.add(Dropout(dropout3))
    model.add(Dense(num_outputs, kernel_initializer=initialiser4, activation=activation4)) # final output has 3 dimensions
    # Print model summary
    model.summary()
    # Compile model
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity'
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])

    return model

