import pandas as pd
import os,sys
import numpy as np
from shared_core.common_functions import make_dir, request_input_path, save_df


def main():
	# take an arbitrary subject for example
	# '/Users/pizarror/mturk/idm_data/split/idm_2022-12-08_12h53.16.781/crdm/idm_2022-12-08_12h53.16.781_crdm.csv'
	CRDM_fn = request_input_path(prompt='Please enter the path to an arbitray CRDM file')
	df = pd.read_csv(CRDM_fn)
	df['crdm_lott_amt'] = df['crdm_lott_top'] + df['crdm_lott_bot']
	df = df.loc[df['crdm_amb_lev']>0].reset_index(drop=True)
	print(df)
	print(list(df))

	# bound 0 < p - beta*A/2 < 1.0; where p = (1-A)/2
	# top bound 0 < p - beta*A/2 : beta < 1/A -1
	df['beta_top'] = 1.0/(df['crdm_amb_lev']/100.0) - 1
	# bottom bound p - beta*A/2 < 1.0 : -1 < beta
	df['beta_bottom'] = -1.0/(df['crdm_amb_lev']/100.0) - 1
	df = df.sort_values(by=['beta_top'])
	df_beta = df[['beta_top','beta_bottom','crdm_amb_lev','crdm_sure_amt', 'crdm_sure_p', 'crdm_lott_amt', 'crdm_lott_p']]
	print(df_beta)

	save_df(df_beta,prompt='Please enter the path where to write beta estimates')


if __name__ == "__main__":
	# main will be executed after running the script
    main()
