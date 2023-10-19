import pandas as pd
import os,sys
import numpy as np
from shared_core.common_functions import make_dir, request_input_path, save_df


def main():
	# take an arbitrary subject for example
	# '/Users/pizarror/mturk/idm_data/split/idm_2022-12-08_12h53.16.781/crdm/idm_2022-12-08_12h53.16.781_crdm.csv'
	CRDM_fn = request_input_path(prompt='Please enter the path to an arbitray CRDM file')
	df = pd.read_csv(CRDM_fn,index_col=0)
	df['crdm_lott_amt'] = df['crdm_lott_top'] + df['crdm_lott_bot']
	df = df.loc[df['crdm_amb_lev']==0].reset_index(drop=True)
	print(df)
	print(list(df))

	# indifference point for each possible set of task values, set SV_now = SV_later to find alpha
	# alpha = log(p_risk) / (log(v_safe/v_risk)) 
	# a = math.log( df['crdm_lott_p'] )
	# b = math.log( df['crdm_sure_amt']/df['crdm_lott_amt'] )
	factor = 1.0
	if np.max(df['crdm_lott_p'])>1.0:
		factor = 100.0
	df['alpha'] = np.log( df['crdm_lott_p']/factor ) / ( np.finfo(np.float32).eps + np.log( df['crdm_sure_amt']/df['crdm_lott_amt'] ) )
	df['log_alpha'] = np.log(df['alpha'])
	df = df.sort_values(by=['alpha'])
	df_alpha = df[['alpha','crdm_sure_amt', 'crdm_sure_p', 'crdm_lott_amt', 'crdm_lott_p','crdm_amb_lev']]
	print(df_alpha)

	save_df(df_alpha,prompt='Please enter the path where to write alpha estimates')


if __name__ == "__main__":
	# main will be executed after running the script
    main()
