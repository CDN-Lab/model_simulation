import os,sys
import numpy as np
import pandas as pd
from shared_core.common_functions import request_path

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
 
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
 
# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from IDM_model.src import CDD_functions

def main():

	# initialize model with a selection of parameters
	gamma0 = 0.702
	kappa0 = 0.017

	CDD_fn = request_path(prompt='Please enter the path to an arbitray CDD file')
	df = pd.read_csv(CDD_fn)
	print(df)

	# CDD_functions



	# bounds for gamma (g) and kappa (k)
	gb = [0,8]
	kb = [1e-3,8]

	# number of samples for each dimension/parameter
	nb_samples = 100

	gamma = np.linspace(gb[0], gb[1], num=nb_samples)
	kappa = np.linspace(kb[0], kb[1], num=nb_samples)



if __name__ == "__main__":
	# main will be executed after running the script
    main()




