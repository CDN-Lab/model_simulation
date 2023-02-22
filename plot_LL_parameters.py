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

from IDM_model.src import model_functions as mf


def plot_save_3D(X,Y,Z,xlabel='',ylabel='',zlabel='',nb_samples=50):

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

	# Plot the surface.
	X, Y = np.meshgrid(X, Y)
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	ax.set_zlabel(zlabel)

	model_sim_dir = '/Users/pizarror/mturk/model_simulation/figs'
	pickle_fn = os.path.join(model_sim_dir,'negLL_{}_{}_{}_samples.fig.pickle'.format(xlabel,ylabel,int(nb_samples**2)))
	# request_save_path(prompt='Please enter the path to the .fig.pickle file to save the plot')
	print('Saving 3D plot to : {}'.format(pickle_fn))
	pickle.dump(fig, open(pickle_fn, 'wb'))

	print('\n\n===Showing the figure in python (or ipython) after saving it===\n\n')
	print('>>> import pickle')
	print(">>> figx = pickle.load(open({}, 'rb'))".format(pickle_fn))
	print('>>> figx.show() for Showing the figure, edit it, etc.!')
	
	plt.show()



def simulate_estimate_CDD_model(fn,gamma0,kappa0,alpha0,verbose=False):

	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['cdd_trial_type']=='task']
	# insert probability as choice into data
	cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait']
	# also returns percent_reward which we do not need here
	data = mf.get_data(df,cols,alpha_hat=alpha0)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice
	prob_choice = mf.probability_choice([gamma0,kappa0],data['cdd_immed_amt'],data['cdd_delay_amt'],
		time_null=data['cdd_immed_wait'],time_reward=data['cdd_delay_wait'],alpha=[alpha0]*df.shape[0])[0]
	data['cdd_trial_resp.corr'] = np.around(np.array(prob_choice))

	# estimate parameters based on self-generated data
	gk_guess = [0.15, 0.5]
	gk_bounds = ((0,8),(0.0022,7.875))
	negLL,gamma_hat,kappa_hat = mf.fit_computational_model(data,guess=gk_guess,bounds=gk_bounds,disp=verbose)
	if verbose:
		print(data)
		print("Negative log-likelihood: {}, gamma: {}, kappa: {}". format(negLL, gamma_hat, kappa_hat))

	return negLL

def simulate_estimate_CRDM_model(fn,gamma0,alpha0,beta0,verbose=False):

	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['crdm_trial_type']=='task']
	# get data with specified columns
	cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
	data = CRDM_functions.get_data(df,cols)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice and insert into data
	prob_choice = CRDM_functions.probability_choose_ambiguity(
		data['crdm_sure_amt'],data['crdm_lott_amt'],data['crdm_sure_p'],data['crdm_lott_p'],data['crdm_amb_lev'],
		[gamma0,beta0,alpha0])[0]
	choice = np.around(np.array(prob_choice))
	data['crdm_trial_resp.corr'] = choice

	# estimate parameters based on self-generated data
	negLL,gamma_hat,beta_hat,alpha_hat = CRDM_functions.fit_ambiguity_risk_model(data,disp=verbose)
	if verbose:
		print(data)
		print("Negative log-likelihood: {}, gamma: {}, beta: {}, alpha: {}". format(negLL, gamma_hat, beta_hat, alpha_hat))

	return negLL




def range_variables(v1_bound,v2_bound,nb_samples=100):

	v1 = np.linspace(v1_bound[0], v1_bound[1], num=nb_samples).tolist()
	v2 = np.linspace(v2_bound[0], v2_bound[1], num=nb_samples).tolist()
	return v1,v2


def simulate_v1_v2(task='CDD',fn='',v1_bound=[0,8],v2_bound=[1e-3,8],v_fixed=1.0,nb_samples=50):
	# nb_samples is number of samples for each variable

	# prepare the variables to range and negLL matrix for storing values
	var1,var2 = range_variables(v1_bound,v2_bound,nb_samples=nb_samples)
	negLL = np.zeros((nb_samples,nb_samples))

	if 'CDD' in task:
		for iv1,v1 in enumerate(var1):
			print(iv1,v1)
			for iv2,v2 in enumerate(var2):
					negLL[iv1,iv2] = simulate_estimate_CDD_model(fn,v1,v2,v_fixed)
	elif 'CRDM' in task:
		for iv1,v1 in enumerate(var1):
			print(iv1,v1)
			for iv2,v2 in enumerate(var2):
					negLL[iv1,iv2] = simulate_estimate_CRDM_model(fn,v1,v2,v_fixed)
	else:
		print('No task selected')

	return var1,var2,negLL


def main():
	# For some reason I cannot run these together, I have to run for one task, save, and rerun script
	nb_samples=50
	task='CDD'

	# /Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/cdd/idm_2022-12-08_14h39.52.884_cdd.csv
	CDD_fn = request_input_path(prompt='Please enter the path to an arbitray CDD file')
	
	# First simulation, fix alpha to 1.0 and vary gamma and kappa
	alpha0 = 1
	# bounds for gamma and kappa
	gamma_bound = [0,8]
	kappa_bound = [0.0022,0.368]
	
	gamma,kappa,negLL = simulate_v1_v2(task=task,fn=CDD_fn,v1_bound=gamma_bound,v2_bound=kappa_bound,v_fixed=alpha0,nb_samples=nb_samples)
	kappa = np.log(kappa)
	plot_save_3D(gamma,kappa,negLL,xlabel='gamma',ylabel='kappa',zlabel='negative log-likelihood',nb_samples=nb_samples)
	'''
	with open('estimates/gkLL.npy', 'wb') as f:
	    np.save(f, gamma)
	    np.save(f, kappa)
	    np.save(f, negLL)


	task='CRDM'

	# /Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/crdm/idm_2022-12-08_14h39.52.884_crdm.csv
	CRDM_fn = request_input_path(prompt='Please enter the path to an arbitray {} file'.format(task))

	# Second simulation, fix alpha to 1.0 and vary gamma and kappa
	beta0 = 0.5
	# bounds for gamma and kappa
	gamma_bound = [0,8]
	alpha_bound = [0.125,4.341]

	gamma,alpha,negLL = simulate_v1_v2(task=task,fn=CRDM_fn,v1_bound=gamma_bound,v2_bound=alpha_bound,v_fixed=beta0,nb_samples=nb_samples)
	with open('estimates/gaLL.npy', 'wb') as f:
	    np.save(f, gamma)
	    np.save(f, alpha)
	    np.save(f, negLL)
	# plot_save_3D(gamma,alpha,negLL,xlabel='gamma',ylabel='alpha',zlabel='negative log-likelihood',nb_samples=nb_samples)
	'''

if __name__ == "__main__":
	# main will be executed after running the script
    main()




