import os,sys
import numpy as np
import pandas as pd
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



def plot_save_3D(fig_idx=1,Xlin=[],Ylin=[],Z=[],xlabel='',ylabel='',zlabel=''):
	# Plot the surface.
	X, Y = np.meshgrid(Xlin, Ylin,indexing='ij')
	fig = plt.figure(fig_idx)
	ax = fig.subplots(subplot_kw={"projection": "3d"})
	ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.5)
	offset = [np.min(Xlin)-1,np.max(Ylin),np.min(np.min(Z))]
	cset = ax.contour(X, Y, Z, zdir='z', offset=offset[2], cmap=cm.coolwarm)
	cset = ax.contour(X, Y, Z, zdir='x', offset=offset[0], cmap=cm.coolwarm)
	cset = ax.contour(X, Y, Z, zdir='y', offset=offset[1], cmap=cm.coolwarm)

	# calc index of min/max Z value
	xmin, ymin = np.unravel_index(np.argmin(Z), Z.shape)
	xmax, ymax = np.unravel_index(np.argmax(Z), Z.shape)

	# min max points in 3D space (x,y,z)
	mi = (X[xmin,ymin], Y[xmin,ymin], Z.min())
	ma = (X[xmax, ymax], Y[xmax, ymax], Z.max())

	# Arrays for plotting, 
	# first row for points in xplane, last row for points in 3D space
	Ami = np.array([mi]*4)
	Ama = np.array([ma]*4)
	for i, v in enumerate(offset):
		Ami[i,i] = v 
		Ama[i,i] = v 

	#plot points.
	ax.plot(Ami[:,0], Ami[:,1], Ami[:,2], marker="o", ls="", c=cm.coolwarm(0.))
	ax.plot(Ama[:,0], Ama[:,1], Ama[:,2], marker="o", ls="", c=cm.coolwarm(1.))
	plt.xlabel(r'${}$'.format(xlabel))
	plt.ylabel(r'${}$'.format(ylabel))
	ax.set_zlabel(r'${}$'.format(zlabel))
	ax.view_init(azim=-45, elev=19)

	fig.tight_layout()
	return plt

def estimate_NLL_model(data,gamma0,beta0,alpha0):
	# estimate parameters based on self-generated data
	parms=[gamma0,beta0,alpha0]
	negLL = mf.function_negLL(parms,data)
	return negLL

def range_variables(v1_bound,v2_bound,nb_samples=100):

	v1 = np.linspace(v1_bound[0], v1_bound[1], num=nb_samples).tolist()
	v2 = np.linspace(v2_bound[0], v2_bound[1], num=nb_samples).tolist()
	return v1,v2

def simulate_data(fn,beta0,gamma0=0.8,alpha0=0.5):
	df = pd.read_csv(fn)
	# remove practice trials
	df = df.loc[df['crdm_trial_type']=='task']
	# insert probability as choice into data
	cols = ['crdm_trial_resp.corr','crdm_sure_amt','crdm_lott_amt','crdm_sure_p','crdm_lott_p','crdm_amb_lev']
	# also returns percent_reward which we do not need here
	data = mf.get_data(df,cols)[0]

	# generate probability based on gamma,kappa then threshold at 0.5 to generate choice
	prob_choose_lott,SV_null,SV_reward = mf.probability_choice([gamma0,beta0,alpha0],data['crdm_sure_amt'],data['crdm_lott_amt'],
		p_null=data['crdm_sure_p'],p_reward=data['crdm_lott_p'],ambiguity=data['crdm_amb_lev'],task='crdm')
	# data['crdm_trial_resp.corr'] = np.around(np.array(prob_choice))

	p_array = np.array(prob_choose_lott)
	# rand_array = np.random.normal(0.0,0.1,p_array.shape)
	bar = np.random.uniform(0,1,p_array.shape)
	# choice = np.around(p_array)#+rand_array)
	choice = p_array > bar
	choice[choice==2]=1 
	choice[choice==-1]=0
	data['crdm_trial_resp.corr'] = choice

	return data

def simulate_v1_v2(fn='',v1_0=0.8,v2_0=0.5,v_fixed=1.0,v1_bound=[0,8],v2_bound=[1e-3,8],nb_samples=50):
	# nb_samples is number of samples for each variable

	# simulate data
	data = simulate_data(fn,beta0=v_fixed,gamma0=v1_0,alpha0=v2_0)
	# prepare the variables to range and negLL matrix for storing values
	var1,var2 = range_variables(v1_bound,v2_bound,nb_samples=nb_samples)
	xsize,ysize = len(var1),len(var2)
	negLL = np.zeros((xsize,ysize))
	for iv1,v1 in enumerate(var1):
		for iv2,v2 in enumerate(var2):
				# v2 = np.exp(v2)
				negLL[iv1,iv2] = estimate_NLL_model(data,v1,v_fixed,v2)
				# print('index ({0},{1}), parms ({2:0.3f},{3:0.3f}), negLL {4:0.3f}'.format(iv1,iv2,v1,v2,negLL[iv1,iv2]))
	return var1,var2,negLL


def save_to_numpy(fn,MSE1,MSE2):
	with open(fn, 'wb') as f:
		np.save(f, MSE1)
		np.save(f, MSE2)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gen_estimates(nb_estimates=10,fn='',var=[0,0],vfix=0.8,v1_bound=[0,8],v2_bound=[0.125,4.341],nb_samples=10):
	hat = [[0,0]]*nb_estimates
	print('\nGround truth for (gamma,alpha): ({0},{1})'.format(var[0],var[1]))
	for i in range(nb_estimates):
		gamma,alpha,negLL = simulate_v1_v2(fn=fn,v1_0=var[0],v2_0=var[1],v_fixed=vfix,v1_bound=v1_bound,v2_bound=v2_bound,nb_samples=nb_samples)
		(row,col) = np.where(negLL == np.min(negLL))
		hat[i] = [gamma[row[0]],alpha[col[0]]]
		print('Min of negLL for (gamma,alpha): ({0:0.3f},{1:0.3f})'.format(hat[i][0],hat[i][1]))
	return hat

def compute_mse(gt,hat):
	mse1 = 0
	mse2 = 0
	for i in range(len(hat)):
		mse1 += (gt[0]-hat[i][0])**2 
		mse2 += (gt[1]-hat[i][1])**2
	# nmse = mse/((gt[0])**2+(gt[1])**2)
	return mse1/len(hat),mse2/len(hat)

def simulate_CRDM(nb_samples=50):
	fn = '/Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/crdm/idm_2022-12-08_14h39.52.884_crdm.csv'
	# CRDM_fn = cf.request_input_path(prompt='Please enter the path to an arbitray {} file'.format(task))
	
	# Second simulation, fix beta to 0.8 and vary gamma and alpha
	# beta_bound = [-4.167,4.167]
	beta0 = 0.8
	# bounds for gamma and alpha
	gamma_bound = [0,8]
	alpha_bound = [0.125,4.341]

	# gamma_bound diminished to not near the edge
	gamma_eps = 0.05*(max(gamma_bound) - min(gamma_bound))
	gamma_bound_dim = [g+(1-2*i)*gamma_eps for i,g in enumerate(gamma_bound)]
	alpha_eps = 0.05*(max(alpha_bound) - min(alpha_bound))
	alpha_bound_dim = [a+(1-2*i)*alpha_eps for i,a in enumerate(alpha_bound)]
	
	nb_var_samples = 5
	gamma_range,alpha_range = range_variables(gamma_bound_dim,alpha_bound_dim,nb_samples=nb_var_samples)

	# figure index for subplot
	fig_idx = 1
	xsize,ysize = len(gamma_range),len(alpha_range)
	MSE_gamma = np.zeros((xsize,ysize))
	NMSE_gamma = np.zeros((xsize,ysize))
	MSE_alpha = np.zeros((xsize,ysize))
	NMSE_alpha = np.zeros((xsize,ysize))

	for ig,gamma0 in enumerate(gamma_range):
		for ia,alpha0 in enumerate(alpha_range):
			# ground truth
			gt = [gamma0,alpha0]
			hat = gen_estimates(nb_estimates=5,fn=fn,var=gt,vfix=beta0,v1_bound=gamma_bound,v2_bound=alpha_bound,nb_samples=nb_samples)
			MSE_gamma[ig,ia],MSE_alpha[ig,ia] = compute_mse(gt,hat)

	fn='estimates/crdm_MSE.npy'
	save_to_numpy(fn,MSE_gamma,MSE_alpha)
	plt = plot_save_3D(fig_idx=1,Xlin=gamma_range,Ylin=alpha_range,Z=MSE_gamma,xlabel=r'\gamma',ylabel=r'\alpha',zlabel=r'MSE_{\gamma}')
	plt = plot_save_3D(fig_idx=2,Xlin=gamma_range,Ylin=alpha_range,Z=MSE_alpha,xlabel=r'\gamma',ylabel=r'\alpha',zlabel=r'MSE_{\alpha}')
	plt.show()

def main():
	nb_samples=50
	simulate_CRDM(nb_samples=nb_samples)


if __name__ == "__main__":
	# main will be executed after running the script
    main()




