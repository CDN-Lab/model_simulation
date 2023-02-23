import os,sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import shared_core.common_functions as cf
# from shared_core.common_functions import request_input_path

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


def plot_save_3D(X,Y,Z,xlabel='',ylabel='',zlabel='',nb_samples=50,verbose=False):

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

	if verbose:
		print('\n\n===Showing the figure in python (or ipython) after saving it===\n\n')
		print('>>> import pickle')
		print(">>> figx = pickle.load(open({}, 'rb'))".format(pickle_fn))
		print('>>> figx.show() for Showing the figure, edit it, etc.!')
	
	plt.show()



def simulate_estimate_CDD_model(index,fn,gamma0,kappa0,alpha0,verbose=False):
	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['cdd_trial_type']=='task']
	# insert probability as choice into data
	cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_immed_wait','cdd_delay_amt','cdd_delay_wait','alpha']
	# also returns percent_reward which we do not need here
	data = mf.get_data(df,cols,alpha_hat=alpha0)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice
	p_choose_reward,SV_null,SV_reward = mf.probability_choice([gamma0,kappa0],data['cdd_immed_amt'],data['cdd_delay_amt'],
		time_null=data['cdd_immed_wait'],time_reward=data['cdd_delay_wait'],alpha=data['alpha'],task='cdd')
	# print(np.around(np.array(prob_choice)))
	p_array = np.array(p_choose_reward)
	rand_array = np.random.normal(0.0,0.2,p_array.shape)
	choice = np.around(p_array+rand_array)
	choice[choice==2]=1
	choice[choice==-1]=0
	data['cdd_trial_resp.corr'] = choice

	# estimate parameters based on self-generated data
	gk_guess = [0.15, 0.005]
	gk_bounds = ((0,6),(0.0022,0.368))
	negLL,gamma_hat,kappa_hat = mf.fit_computational_model(data,guess=gk_guess,bounds=gk_bounds,disp=verbose)
	print('Kappa Hat : {}'.format(kappa_hat))

	# sorted for plotting
	SV_delta = [rew-null for (rew,null) in zip(SV_reward,SV_null)]	
	SV_delta, p_choose_reward = zip(*sorted(zip(SV_delta, p_choose_reward)))
	plt = mf.plot_fit(index,SV_delta,p_choose_reward,choice=data['cdd_trial_resp.corr'].tolist(),ylabel='prob_choose_delay',xlabel='SV difference (SV_delay - SV_immediate)',
		title=r'$\gamma={0:0.3f}, \kappa={1:0.3f}$'.format(gamma0,kappa0))
	print('Kappa Hat : {}'.format(kappa_hat))
	textstr = r'$(\hat \gamma,\hat \kappa) : ({0:0.3f},{1:0.3f})$'.format(gamma_hat,kappa_hat)
	# these are matplotlib.patch.Patch properties
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	# place a text box in upper left in axes coords
	xpos = 0.5*(max(SV_delta) - min(SV_delta)) + min(SV_delta)
	plt.text(xpos, 0.8, textstr, fontsize=14,verticalalignment='top', bbox=props)

	model_sim_dir = '/Users/pizarror/mturk/model_simulation/figs/choice_fit'	
	cf.make_dir(model_sim_dir)
	fig_fn = os.path.join(model_sim_dir,'gamma_{0:0.4f}_kappa_{1:0.4f}.png'.format(gamma0,kappa0))
	plt.savefig(fig_fn)
	print('Saving to : {}'.format(fig_fn))
	plt.close(index)

	print('Kappa Hat : {}'.format(kappa_hat))
	print('Ground truth for (gamma,kappa) : ({},{})'.format(gamma0,kappa0))
	print('Estimated values (gamma,kappa) : ({},{})'.format(gamma_hat,kappa_hat))
	if verbose:
		print(data)
		print("Negative log-likelihood: {}, gamma: {}, kappa: {}". format(negLL, gamma_hat, kappa_hat))
	return negLL,gamma_hat,kappa_hat

def simulate_estimate_CRDM_model(index,fn,gamma0,alpha0,beta0,verbose=False):
	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['crdm_trial_type']=='task']
	# get data with specified columns
	cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
	data = mf.get_data(df,cols)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice and insert into data
	prob_choice = mf.probability_choice([gamma0,beta0,alpha0],data['crdm_sure_amt'],data['crdm_lott_amt'],
		p_null=data['crdm_sure_p'],p_reward=data['crdm_lott_p'],ambiguity=data['crdm_amb_lev'],task='crdm')[0]
	data['crdm_trial_resp.corr'] = np.around(np.array(prob_choice))

	# estimate parameters based on self-generated data
	gba_guess = [0.15, 0.5, 0.6]
	gba_bounds = ((0,8),(1e-8,6.4),(0.125,4.341))
	negLL,gamma_hat,beta_hat,alpha_hat = mf.fit_computational_model(data,guess=gba_guess,bounds=gba_bounds,disp=verbose)
	print('Ground truth for (gamma,alpha,beta) : ({},{},{})'.format(gamma0,alpha0,beta0))
	print('Estimated values (gamma,alpha,beta) : ({},{},{})'.format(gamma_hat,alpha_hat,beta_hat))
	if verbose:
		print(data)
		print("Negative log-likelihood: {}, gamma: {}, beta: {}, alpha: {}". format(negLL, gamma_hat, beta_hat, alpha_hat))
	return negLL


def range_variables(v1_bound,v2_bound,nb_samples=100):

	v1 = np.linspace(v1_bound[0], v1_bound[1], num=nb_samples).tolist()
	v2 = np.linspace(v2_bound[0], v2_bound[1], num=2*nb_samples).tolist()
	return v1,v2


def plot_ground_hat(v1_ground,v2_ground,v1_hat,v2_hat):
	plt.figure(1000)
	print(v1_hat)
	for i in range(v2_ground.shape[1]):
		plt.plot(v1_ground[:,i],v1_hat[:,i],'*-',label=r'$\kappa = {0:0.3f}$'.format(v2_ground[3,i]))
		plt.xlabel(r'$\gamma_{truth}$',fontsize=12)
		plt.ylabel(r'$\gamma_{estimate}$',fontsize=12)
	plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	plt.figure(1001)
	print(v2_hat)
	for i in range(v1_ground.shape[0]):
		plt.plot(np.log(v2_ground[i,:]),np.log(v2_hat[i,:]),'*-',label=r'$\gamma = {0:0.3f}$'.format(v1_ground[i,0]))
		plt.xlabel(r'$\kappa_{truth}$',fontsize=12)
		plt.ylabel(r'$\kappa_{estimate}$',fontsize=12)
	plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
	plt.tight_layout()
	plt.show()
	sys.exit()


def simulate_v1_v2(task='CDD',fn='',v1_bound=[0,8],v2_bound=[1e-3,8],v_fixed=1.0,nb_samples=50):
	# nb_samples is number of samples for each variable

	# prepare the variables to range and negLL matrix for storing values
	var1,var2 = range_variables(v1_bound,v2_bound,nb_samples=nb_samples)
	xsize,ysize = len(var1),len(var2)
	negLL = np.zeros((xsize,ysize))
	v1_ground = np.zeros((xsize,ysize))
	v1_hat = np.zeros((xsize,ysize))
	v2_ground = np.zeros((xsize,ysize))
	v2_hat = np.zeros((xsize,ysize))
	index = 0
	if 'CDD' in task:
		for iv1,v1 in enumerate(var1):
			print(iv1,v1)
			for iv2,v2 in enumerate(var2):
					print(iv2,v2)
					v2 = np.exp(v2)
					print(iv1,iv2,v1)
					print(iv1,iv2,v2)
					v1_ground[iv1,iv2] = v1
					v2_ground[iv1,iv2] = v2
					negLL[iv1,iv2],v1_hat[iv1,iv2],v2_hat[iv1,iv2] = simulate_estimate_CDD_model(index,fn,v1,v2,v_fixed)
					index += 1
		plot_ground_hat(v1_ground,v2_ground,v1_hat,v2_hat)
	elif 'CRDM' in task:
		for iv1,v1 in enumerate(var1):
			print(iv1,v1)
			for iv2,v2 in enumerate(var2):
					negLL[iv1,iv2] = simulate_estimate_CRDM_model(index,fn,v1,v2,v_fixed)
					index += 1
	else:
		print('No task selected')

	return var1,var2,negLL


def save_to_numpy(fn,gamma,kappa,negLL):
	with open(fn, 'wb') as f:
	    np.save(f, gamma)
	    np.save(f, kappa)
	    np.save(f, negLL)

def simulate_CDD(nb_samples=50):
	task='CDD'

	CDD_fn = '/Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/cdd/idm_2022-12-08_14h39.52.884_cdd.csv'
	# CDD_fn = cf.request_input_path(prompt='Please enter the path to an arbitray CDD file')
	
	# First simulation, fix alpha to 1.0 and vary gamma and kappa
	alpha0 = 1
	# bounds for gamma and kappa : (noise and discount rate)
	gamma_bound = [0,5]
	# range for ln(discount_rate) : [-6,-1]
	# kappa_bound = [0.0022,0.368]
	log_discount_rate_bound = [-6,-1]
	
	gamma,kappa,negLL = simulate_v1_v2(task=task,fn=CDD_fn,v1_bound=gamma_bound,v2_bound=log_discount_rate_bound,v_fixed=alpha0,nb_samples=nb_samples)
	# kappa = np.log(kappa)
	plot_save_3D(gamma,kappa,negLL,xlabel='gamma',ylabel='kappa',zlabel='negative log-likelihood',nb_samples=nb_samples,verbose=False)

	# fn='estimates/kaLL.npy'
	# save_to_numpy(fn,gamma,kappa,negLL)

def simulate_CRDM(nb_samples=50):
	task='CRDM'

	CRDM_fn = '/Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/crdm/idm_2022-12-08_14h39.52.884_crdm.csv'
	# CRDM_fn = cf.request_input_path(prompt='Please enter the path to an arbitray {} file'.format(task))

	# Second simulation, fix alpha to 1.0 and vary gamma and kappa
	beta0 = 0.8
	# bounds for gamma and alpha
	gamma_bound = [0,8]
	alpha_bound = [0.125,4.341]

	gamma,alpha,negLL = simulate_v1_v2(task=task,fn=CRDM_fn,v1_bound=gamma_bound,v2_bound=alpha_bound,v_fixed=beta0,nb_samples=nb_samples)
	plot_save_3D(gamma,alpha,negLL,xlabel='gamma',ylabel='alpha',zlabel='negative log-likelihood',nb_samples=nb_samples,verbose=False)
	
	# fn='estimates/gaLL.npy'
	# save_to_numpy(fn,gamma,kappa,negLL)
	
def main():
	# For some reason I cannot run these together, I have to run for one task, save, and rerun script
	nb_samples=5

	simulate_CDD(nb_samples=nb_samples)
	# simulate_CRDM(nb_samples=nb_samples)



if __name__ == "__main__":
	# main will be executed after running the script
    main()




