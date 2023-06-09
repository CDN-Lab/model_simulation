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


def plot_save_3D(Xlin,Ylin,Z,gt,c0,c_hat,fig_info = (0,0),xlabel='',ylabel='',zlabel='',nb_samples=50,verbose=False):
	print('coordinates of ground truth : {}'.format(c0))
	print('coordinates of estimate : {}'.format(c_hat))
	# Plot the surface.
	X, Y = np.meshgrid(Xlin, Ylin,indexing='ij')
	print('ground truth')
	print(Xlin[c0[0]], Ylin[c0[1]], Z[c0[0],c0[1]])
	print('estimate')
	print(Xlin[c_hat[0]], Ylin[c_hat[1]], Z[c_hat[0],c_hat[1]])

	fig = plt.figure(1001)
	# ax = fig.gca(projection='3d')
	# fig, ax = plt.subplots(fig_info[0],fig_info[0],fig_info[1], subplot_kw={"projection": "3d"})
	ax = fig.add_subplot(fig_info[0],fig_info[0],fig_info[1], projection='3d')
	ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
	# ax.scatter(X[c0[0],c0[1]], Y[c0[0],c0[1]], Z[c0[0],c0[1]], c='green', marker='^', s=100)
	ax.scatter(gt[0], gt[1], 0, c='green', marker='^', s=100)
	# ax.scatter(X[c_hat[0],c_hat[1]], Y[c_hat[0],c_hat[1]], Z[c_hat[0],c_hat[1]], c='black', marker='*', s=1000)
	# cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(np.min(Z))-100, cmap=cm.coolwarm)
	cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(Xlin), cmap=cm.coolwarm)
	cset = ax.contour(X, Y, Z, zdir='y', offset=np.max(Ylin), cmap=cm.coolwarm)

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
	for i, v in enumerate([np.min(Xlin),np.max(Ylin)]):
		Ami[i,i] = v 
		Ama[i,i] = v 

	#plot points.
	ax.plot(Ami[:,0], Ami[:,1], Ami[:,2], marker="o", ls="", c=cm.coolwarm(0.))
	# ax.plot(Ama[:,0], Ama[:,1], Ama[:,2], marker="o", ls="", c=cm.coolwarm(1.))
	if (fig_info[1]-1)//fig_info[0] == (fig_info[0]-1):
		plt.xlabel(r'${}$'.format(xlabel))
		plt.ylabel(r'${}$'.format(ylabel))
	if (fig_info[1] <= fig_info[0]):
		plt.title(r'${0}$ : {1:0.3f}'.format(ylabel,gt[1]))
	# plt.title(r'${0}$,${1}$ :: {2:0.3f},{3:0.3f}'.format(xlabel,ylabel,gt[0],np.log(gt[1])))
	if not np.mod(fig_info[1],fig_info[0]):
		ax.set_zlabel(r'{0}, ${1}$ : {2:0.3f}'.format(zlabel,xlabel,gt[0]))
	ax.view_init(azim=-45, elev=19)

	return fig,plt

""" 
	fn = 'figs/negLL_gamma_kappa/negLL_gamma_{0:0.3f}_logkappa_{1:0.3f}.eps'.format(gt[0],np.log(gt[1]))
	print('Saving to : {}'.format(fn))
	plt.savefig(fn,format='eps')
	plt.close()
 """

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


def save_to_numpy(fn,gamma,kappa,negLL):
	with open(fn, 'wb') as f:
		np.save(f, gamma)
		np.save(f, kappa)
		np.save(f, negLL)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def simulate_CRDM(nb_samples=50):
	CRDM_fn = '/Users/pizarror/mturk/idm_data/batch_output/bonus2/idm_2022-12-08_14h39.52.884/crdm/idm_2022-12-08_14h39.52.884_crdm.csv'
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

	# gamma_range = [0.4, 2.0, 2.8, 4.4, 7.6]
	# alpha_range = [0.336,0.757, 3.28, 3.71,4.13] #1.6, 2.86,

	# figure index for subplot
	fig_idx = 1
	for gamma0 in gamma_range:
		for alpha0 in alpha_range:
			# ground truth
			gt = [gamma0,alpha0]
			print(gt)

			gamma,alpha,negLL = simulate_v1_v2(fn=CRDM_fn,v1_0=gamma0,v2_0=alpha0,v_fixed=beta0,v1_bound=gamma_bound,v2_bound=alpha_bound,nb_samples=nb_samples)
			(row,col) = np.where(negLL == np.min(negLL))
			row0 = find_nearest(gamma,gamma0)
			col0 = find_nearest(alpha,alpha0)
			coords0 = (row0,col0)
			print('Ground truth for (gamma,alpha): ({0},{1})'.format(gamma0,alpha0))
			print('Min of negLL for (gamma,alpha): ({0:0.3f},{1:0.3f})'.format(gamma[row[0]],alpha[col[0]]))
			coords_hat = (row[0],col[0])
			fig,plt = plot_save_3D(gamma,alpha,negLL,gt,coords0,coords_hat,fig_info=(nb_var_samples,fig_idx), xlabel='\gamma',ylabel=r'\alpha',zlabel='NLL',nb_samples=nb_samples,verbose=False)
			fig_idx += 1
			# fn='estimates/crdm_gkLL.npy'
			# save_to_numpy(fn,gamma,kappa,negLL)
	fig.tight_layout()
	fn = 'figs/negLL_gamma_alpha.eps'
	print('Saving to : {}'.format(fn))
	plt.savefig(fn,format='eps')
	plt.show()

	
def main():
	# For some reason I cannot run these together, I have to run for one task, save, and rerun script
	nb_samples=50
	simulate_CRDM(nb_samples=nb_samples)


if __name__ == "__main__":
	# main will be executed after running the script
    main()




