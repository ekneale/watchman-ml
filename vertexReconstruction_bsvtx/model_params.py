
# This is where parameters are defined to avoid and
# should be the only place with hard-coded variables!

num_outputs = 3
num_features = 5
max_hits = 500

# singles with bs vtx
activation1 = 'linear'
#activation2 = 'relu'
#activation3 = 'linear'
activation4 = 'linear'

initialiser1 = 'he_uniform'
#initialiser2 = 'uniform'
#initialiser3 = 'normal'
initialiser4 = 'he_uniform'

neurons1 = 256
#neurons1 = 476
#neurons2 = 351
#neurons3 = 43

batchsize = 64
num_epochs = 128
#batchsize = 51
#num_epochs = 193

'''
# singles without bs vertex
batchsize = 105
num_epochs = 228

neurons1 = 435
neurons2 = 380
neurons3 = 125

#dropout1 = 0.16041859573407766
#dropout2 = 0.1834787610015903
#dropout3 = 0.04567764468677462

activation1 = 'linear'
activation2 = 'linear'
activation3 = 'relu'
activation4 = 'linear'

initialiser1 = 'uniform'
initialiser2 = 'he_normal'
initialiser3 = 'he_uniform'
initialiser4 = 'he_normal'
'''

