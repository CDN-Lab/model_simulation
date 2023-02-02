import os,sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
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


def simulate_estimate_model(fn,gamma0,kappa0,alpha0,verbose=False):

	df = pd.read_csv(fn)
	df = df.loc[df['cdd_trial_type']=='task']

	# generate probability based on gamma,kappa and threshold at 0.5
	p_choose_delay = CDD_functions.probability_choose_delay(
		df['cdd_immed_amt'],df['cdd_immed_wait'],df['cdd_delay_amt'],df['cdd_delay_wait'],
		[gamma0,kappa0],[alpha0]*df.shape[0])[0]
	prob_array = np.around(np.array(p_choose_delay))

	# insert probability as choice into data_choice_amt_wait
	cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait']
	data_choice_amt_wait = CDD_functions.get_data(df,cols,alpha_hat=alpha0)[0]
	data_choice_amt_wait['cdd_trial_resp.corr'] = prob_array

	# estimate parameters based on self-generated data
	gk_bounds = ((0,8),(1e-3,8))
	negLL,gamma_hat,kappa_hat = CDD_functions.fit_delay_discount_model(data_choice_amt_wait,
		gk_guess = [0.15, 0.5],gk_bounds = gk_bounds, disp=verbose)
	if verbose:
		print(data_choice_amt_wait)
		print("Negative log-likelihood: {}, gamma: {}, kappa: {}". format(negLL, gamma_hat, kappa_hat))

	return negLL


def range_variables(v1_bound,v2_bound,nb_samples=100):

	v1 = np.linspace(v1_bound[0], v1_bound[1], num=nb_samples).tolist()
	v2 = np.linspace(v2_bound[0], v2_bound[1], num=nb_samples).tolist()
	return v1,v2


def main():

	CDD_fn = request_path(prompt='Please enter the path to an arbitray CDD file')
	
	# initialize model with a selection of parameters
	gamma0 = 0.702
	kappa0 = 0.017
	alpha0=1

	# bounds for gamma and kappa
	gamma_bound = [0,8]
	kappa_bound = [1e-3,8]
	# for each variable
	nb_samples = 50
	gamma,kappa = range_variables(gamma_bound,kappa_bound,nb_samples=nb_samples)
	negLL = np.zeros((nb_samples,nb_samples))

	for ig,g in enumerate(gamma):
		print(ig,g)
		for ik,k in enumerate(kappa):
			inegLL = simulate_estimate_model(CDD_fn,g,k,alpha0)
			negLL[ig,ik] = inegLL
	print(negLL)

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Plot the surface.
	gamma, kappa = np.meshgrid(gamma, kappa)
	surf = ax.plot_surface(gamma, kappa, negLL, cmap=cm.coolwarm,
		linewidth=0, antialiased=False)
	plt.xlabel('gamma')
	plt.ylabel('kappa')
	ax.set_zlabel('negative log-likelihood')

	pickle_fn = '/Users/pizarror/mturk/figs/model_simulation/negLL_gamma_kappa_2500_samples.fig.pickle'
	print('Saving 3D plot to : {}'.format(pickle_fn))
	pickle.dump(fig, open(pickle_fn, 'wb'))
	
	plt.show()

	'''
	===Showing the figure after saving it===
	import pickle
	figx = pickle.load(open(pickle_fn, 'rb'))
	figx.show() # Show the figure, edit it, etc.!
	'''

if __name__ == "__main__":
	# main will be executed after running the script
    main()




