from model_params import *

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout, Concatenate, Input


def model():
    '''
    Model building function.
    '''
    # Clear the previous model to free up GPU space 
    keras.backend.clear_session()
    
    # first input model
    hits = Input(shape=(num_features,max_hits))
    mask = Masking(mask_value=0.)(hits)
    flat1 = Flatten()(mask)

    # second input model
    bonsai_vtx = Input(shape=(3))
#    flat2 = Flatten()(bonsai_vtx)


    # interpretation model
    hidden1 = Dense(neurons1, kernel_initializer=initialiser1, activation=activation1)(flat1)
#    model.add(Dropout(dropout1))
#    hidden2 = Dense(neurons2, kernel_initializer=initialiser2, activation=activation2)(hidden1)
#    model.add(Dropout(dropout2))
#    hidden3 = Dense(neurons3, kernel_initializer=initialiser3, activation=activation3)(hidden2)
#    model.add(Dropout(dropout3))
    # merge input models
    merge = Concatenate()([hidden1,bonsai_vtx])
    output = Dense(num_outputs, kernel_initializer=initialiser4, activation=activation4)(merge)
    model = Model(inputs=[hits,bonsai_vtx],outputs=output)
    # Print model summary
    print(model.summary())
    # Compile model
    # 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'cosine_similarity'
    model.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['accuracy'])

    return model

