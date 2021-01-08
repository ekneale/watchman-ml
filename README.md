Machine learning projects for WATCHMAN:

vertex_reconstruction:

Performs reconstruction of interaction vertex for pre-selected pairs of events using keras.

1. get_features_pair.cc extracts hit features for pairs of events from raw rat simulation data
2. vertex_reconstruction_keras_optimise_hyperas.py performs simultaneous optimisation of hyperparameters with hyperas
3. vertex_reconstruction_keras_optimise_gridsearch.py performs optimisation of individual hyperparameters
4. vertex_reconstruction_keras_train.py constructs a model from the training data
5. vertex_reconstruction_keras_predict.py makes predictions for the interaction vertex in x, y and z

Additional requirement: ROOT for vertex_reconstruction_keras_predict.py

To view the training and validation statistics graphically, run:
tensorboard --logdir logs/train after the training stage and open the link generated.
