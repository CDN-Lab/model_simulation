import os,sys
import numpy as np


# initialize model with a selection of parameters
gamma0 = 






# bounds for gamma (g) and kappa (k)
gb = [0,8]
kb = [1e-3,8]

# number of samples for each dimension/parameter
nb_samples = 100

gamma = np.linspace(gb[0], gb[1], num=nb_samples)
kappa = np.linspace(kb[0], kb[1], num=nb_samples)






