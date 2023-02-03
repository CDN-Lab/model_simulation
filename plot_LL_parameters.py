import os,sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from shared_core.common_functions import request_input_path

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
	# remove practice trials
	df = df.loc[df['cdd_trial_type']=='task']

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice
	prob_choice = CDD_functions.probability_choose_delay(
		df['cdd_immed_amt'],df['cdd_immed_wait'],df['cdd_delay_amt'],df['cdd_delay_wait'],
		[gamma0,kappa0],[alpha0]*df.shape[0])[0]
	choice = np.around(np.array(prob_choice))

	# insert probability as choice into data_choice_amt_wait
	cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait']
	data_choice_amt_wait = CDD_functions.get_data(df,cols,alpha_hat=alpha0)[0]
	data_choice_amt_wait['cdd_trial_resp.corr'] = choice

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

def plot_save_3D(X,Y,Z,xlabel='',ylabel='',zlabel='',nb_samples=50):

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Plot the surface.
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ax.set_zlabel(zlabel)

	model_sim_dir = '/Users/pizarror/mturk/figs/model_simulation/'
	pickle_fn = os.path.join(model_sim_dir,'negLL_{}_{}_{}_samples.fig.pickle'.format(xlabel,ylabel,int(nb_samples**2)))
	# request_save_path(prompt='Please enter the path to the .fig.pickle file to save the plot')
	print('Saving 3D plot to : {}'.format(pickle_fn))
	pickle.dump(fig, open(pickle_fn, 'wb'))
	
	plt.show()


def simulate_gamma_kappa(fn,nb_samples=50):
	# set alpha=1, simulate over other variables
	alpha0 = 1

	# bounds for gamma and kappa
	gamma_bound = [0,8]
	kappa_bound = [1e-3,8]

	# number of samples for each variable
	gamma,kappa = range_variables(gamma_bound,kappa_bound,nb_samples=nb_samples)
	negLL = np.zeros((nb_samples,nb_samples))

	for ig,g in enumerate(gamma):
		print(ig,g)
		for ik,k in enumerate(kappa):
			inegLL = simulate_estimate_model(fn,g,k,alpha0)
			negLL[ig,ik] = inegLL

	return gamma,kappa,negLL


def main():

	CDD_fn = request_input_path(prompt='Please enter the path to an arbitray CDD file')
	
	nb_samples=50
	gamma,kappa,negLL = simulate_gamma_kappa(CDD_fn,nb_samples=nb_samples)
	plot_save_3D(gamma,kappa,negLL,xlabel='gamma',ylabel='kappa',zlabel='negative log-likelihood',nb_samples=nb_samples)


	'''
	===Showing the figure after saving it===
	import pickle
	figx = pickle.load(open(pickle_fn, 'rb'))
	figx.show() # Show the figure, edit it, etc.!
	'''

if __name__ == "__main__":
	# main will be executed after running the script
    main()




