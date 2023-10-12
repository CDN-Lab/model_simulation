#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------------

### Parameters Visualization ###

This script loads the analysis output files generated by IDM_model and creates plots
to visualize the distribution. We will use the distribution of these parameters to 
inform our selection of parameters in the model_simulation project 

Inputs: 
  	user-specified paths to analysis spreadsheets that result from IDM_model, e.g.: 
  		/Users/pizarror/mturk/idm_data/batch_output/bonus2/bonus2_CDD_analysis.csv
  		/Users/pizarror/mturk/idm_data/batch_output/bonus2/bonus2_CDD_analysis_alpha.csv
  		/Users/pizarror/mturk/idm_data/batch_output/bonus2/bonus2_CRDM_analysis.csv
  		/Users/pizarror/mturk/idm_data/batch_output/SDAN/SDAN_CRDM_analysis.csv

Outputs: 
  	plots of the distribution of the each parameter. For CDD, we will have plots, 
  		likely histograms, for gamma (inverse temperature), kappa (discount rate).
  		For CRDM, we will have histogram plots for alpha (risk aversion), beta 
  		(ambiguity aversion), and gamma (inverse temperature).

Usage: $ python parameters_visualization.py

--------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import os,sys

# Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Own modules
import shared_core.common_functions as cf


__author__ = 'Ricardo Pizarro'
__copyright__ = 'Copyright 2023, Introspection and decision-making (IDM) project'
__credits__ = ['Ricardo Pizarro, Silvia Lopez-Guzman']
__license__ = 'IDM_model 1.0'
__version__ = '0.1.0'
__maintainer__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarro@nih.gov, silvia.lopezguzman@nih.gov'
__status__ = 'Dev'



def scatter_hist(x, y, ax, ax_histx, ax_histy,xlabel='',ylabel='',title=''):
	ax.scatter(x, y)
	ax.set_xlabel(xlabel,fontsize=14)
	ax.set_ylabel(ylabel,fontsize=14)
	plt.suptitle(title,fontsize=20)
	ax_histx.hist(x,bins=20)
	ax_histy.hist(y, bins=20,orientation = 'horizontal')


def setup_fig_ax():

	fig = plt.figure()
	gs = GridSpec(4, 4)

	ax = fig.add_subplot(gs[1:4, 0:3])
	ax_histx = fig.add_subplot(gs[0,0:3])
	ax_histy = fig.add_subplot(gs[1:4, 3])
	ax_histx.tick_params(axis="x", labelbottom=False)
	ax_histy.tick_params(axis="y", labelleft=False)

	return ax,ax_histx,ax_histy

def print_stats(x,y,x_name='gamma',y_name='log_kappa'):
	print('{0} (mean,var):({1:0.2f},{2:0.2f})'.format(x_name,np.mean(x),np.var(x)))
	print('{0} (mean,var):({1:0.2f},{2:0.2f})'.format(y_name,np.mean(y),np.var(y)))


def CDD_plots(fn=''):
	# Can also ask for the path using the following:
	# fn = cf.request_input_path(prompt='Please enter the path to an arbitray {} file'.format(task))
	df = pd.read_csv(fn,index_col=0)
	# filter out modeled data at the boundary
	# df = df.loc[(df['kappa']<7) & (df['kappa']>0.01)]
	# df = df.loc[(df['gamma']<5)]

	# kappa versus gamma
	x,y = df['gamma'],np.log(df['kappa'])
	print_stats(x,y,x_name='gamma',y_name='log_kappa')
	print_stats(x,y[y>np.log(0.0022)],x_name='gamma',y_name='log_kappa')
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$\gamma$ - inverse temperate (noise)',
		ylabel=r'$\log \kappa$ - discount rate',
		title=os.path.basename(fn).replace('.csv',''))

	# R^2 versus gamma
	x,y = df['gamma'],df['R2']
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$\gamma$ - inverse temperate (noise)',
		ylabel=r'$R^2$ - coefficient of determination',
		title=os.path.basename(fn).replace('.csv',''))

	# kappa versus R^2
	x,y = df['R2'],np.log(df['kappa'])
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$R^2$ - coefficient of determination',
		ylabel=r'$\log \kappa$ - discount rate',
		title=os.path.basename(fn).replace('.csv',''))


def CRDM_plots(fn=''):

	# fn = '/Users/pizarror/mturk/idm_data/batch_output/raw/raw_CRDM_analysis.csv'
	# Can also ask for the path using the following:
	# fn = cf.request_input_path(prompt='Please enter the path to an arbitray {} file'.format(task))
	df = pd.read_csv(fn,index_col=0)
	# filter out modeled data at the boundary
	df = df.loc[(df['alpha']>0.125)]
	df = df.loc[(df['gamma']<5)]

	# alpha versus gamma
	x,y = df['gamma'],np.log(df['alpha'])
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$\gamma$ - inverse temperate (noise)',
		ylabel=r'$\log \alpha$ - risk attitude',
		title=os.path.basename(fn).replace('.csv',''))

	# alpha versus beta
	x,y = np.log(df['beta']),np.log(df['alpha'])
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$\log \beta$ - ambiguity aversion',
		ylabel=r'$\log \alpha$ - risk attitude',
		title=os.path.basename(fn).replace('.csv',''))

	# R^2 versus gamma
	x,y = df['gamma'],df['R2']
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$\gamma$ - inverse temperate (noise)',
		ylabel=r'$R^2$ - coefficient of determination',
		title=os.path.basename(fn).replace('.csv',''))

	# alpha versus R^2
	x,y = df['R2'],np.log(df['alpha'])
	ax,ax_histx,ax_histy = setup_fig_ax()
	scatter_hist(x,y,ax,ax_histx,ax_histy, 
		xlabel=r'$R^2$ - coefficient of determination',
		ylabel=r'$\log \alpha$ - risk attitude',
		title=os.path.basename(fn).replace('.csv',''))


def main():


	print('Welcome to parameters visualization, lets get started')
	# We will start with CDD_analysis.csv for now
	fn = '/Volumes/UCDN/datasets/IDM/utility/split_CDD_analysis.csv'
	CDD_plots(fn)

	fn = '/Volumes/UCDN/datasets/IDM/utility/split_CDD_analysis_alpha.csv'
	# CDD_plots(fn)
	
	fn = '/Volumes/UCDN/datasets/IDM/utility/split_CRDM_analysis.csv'
	# CRDM_plots(fn)

	plt.show()


if __name__ == "__main__":
	# main will be executed after running the script
    main()




