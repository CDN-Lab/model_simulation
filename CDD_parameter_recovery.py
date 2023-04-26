import os,sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import shared_core.common_functions as cf

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



def estimate_NLL_CDD_model(data,gamma0,kappa0):
	# estimate parameters based on self-generated data
	parms=[gamma0,kappa0]
	negLL = mf.function_negLL(parms,data)
	return negLL


def range_variables(v1_bound,v2_bound,nb_samples=100):

	v1 = np.linspace(v1_bound[0], v1_bound[1], num=nb_samples).tolist()
	v2 = np.linspace(v2_bound[0], v2_bound[1], num=nb_samples).tolist()
	return v1,v2


def plot_ground_hat(v1_ground,v2_ground,v1_hat,v2_hat):

	plt.figure(1000)
	print(v1_hat)
	for i in range(v2_ground.shape[1]):
		plt.plot(v1_ground[:,i],v1_hat[:,i],'*-',label=r'$\alpha = {0:0.3f}$'.format(v2_ground[3,i]))
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

def simulate_data(fn,alpha0,gamma0,kappa0):
	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['cdd_trial_type']=='task']
	# insert probability as choice into data
	cols = ['cdd_trial_resp.corr','cdd_immed_amt','cdd_delay_amt','cdd_immed_wait','cdd_delay_wait','alpha']
	# also returns percent_reward which we do not need here
	data = mf.get_data(df,cols,alpha_hat=alpha0)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice
	p_choose_reward,SV_null,SV_reward = mf.probability_choice([gamma0,kappa0],data['cdd_immed_amt'],data['cdd_delay_amt'],
		time_null=data['cdd_immed_wait'],time_reward=data['cdd_delay_wait'],alpha=data['alpha'],task='cdd')
	# print(np.around(np.array(prob_choice)))

	p_array = np.array(p_choose_reward)
	# rand_array = np.random.normal(0.0,0.1,p_array.shape)
	bar = np.random.uniform(0,1,p_array.shape)
	# choice = np.around(p_array)#+rand_array)
	choice = p_array > bar
	choice[choice==2]=1 
	choice[choice==-1]=0
	data['cdd_trial_resp.corr'] = choice

	return data

def simulate_v1_v2(fn='',v1_bound=[0,8],v2_bound=[1e-3,8],v_fixed=1.0,nb_samples=50):
	# nb_samples is number of samples for each variable

	# simulate data
	data = simulate_data(fn,alpha0=1,gamma0=0.8,kappa0=0.5)

	# prepare the variables to range and negLL matrix for storing values
	var1,var2 = range_variables(v1_bound,v2_bound,nb_samples=nb_samples)
	xsize,ysize = len(var1),len(var2)
	negLL = np.zeros((xsize,ysize))
	v1_ground = np.zeros((xsize,ysize))
	v1_hat = np.zeros((xsize,ysize))
	v2_ground = np.zeros((xsize,ysize))
	v2_hat = np.zeros((xsize,ysize))
	index = 0
	for iv1,v1 in enumerate(var1):
		print(iv1,v1)
		for iv2,v2 in enumerate(var2):
				print(iv2,v2)
				v2 = np.exp(v2)
				print(iv1,iv2,v1)
				print(iv1,iv2,v2)
				v1_ground[iv1,iv2] = v1
				v2_ground[iv1,iv2] = v2
				negLL[iv1,iv2],v1_hat[iv1,iv2],v2_hat[iv1,iv2] = estimate_NLL_CDD_model(index,data,v1,v2,v_fixed)
				index += 1
	plot_ground_hat(v1_ground,v2_ground,v1_hat,v2_hat)

	return var1,var2,negLL


def save_to_numpy(fn,gamma,kappa,negLL):
	with open(fn, 'wb') as f:
		np.save(f, gamma)
		np.save(f, kappa)
		np.save(f, negLL)

def simulate_CDD(nb_samples=50):
	CDD_fn = '/Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/cdd/idm_2022-12-08_14h39.52.884_cdd.csv'
	# CDD_fn = cf.request_input_path(prompt='Please enter the path to an arbitray CDD file')
	
	# First simulation, fix alpha to 1.0 and vary gamma and kappa
	alpha0 = 1
	# bounds for gamma and kappa : (noise and discount rate)
	gamma_bound = [0,8]
	# choice set space kappa = [0.0022,7.8750]
	# range for ln(discount_rate) : [-6,-1]
	# kappa_bound = [0.0022,0.368]
	log_discount_rate_bound = [-8,1]
	
	gamma,kappa,negLL = simulate_v1_v2(fn=CDD_fn,v1_bound=gamma_bound,v2_bound=log_discount_rate_bound,v_fixed=alpha0,nb_samples=nb_samples)
	# kappa = np.log(kappa)
	# plot_save_3D(gamma,kappa,negLL,xlabel='gamma',ylabel='kappa',zlabel='negative log-likelihood',nb_samples=nb_samples,verbose=False)

	fn='estimates/cdd_gkLL.npy'
	save_to_numpy(fn,gamma,kappa,negLL)

	
def main():
	# For some reason I cannot run these together, I have to run for one task, save, and rerun script
	nb_samples=50

	simulate_CDD(nb_samples=nb_samples)


if __name__ == "__main__":
	# main will be executed after running the script
    main()




