import pandas as pd
import numpy as np
import os,sys
from shared_core.common_functions import make_dir, request_input_path, save_df


def main():
	# take an arbitrary subject for example
	# '/Users/pizarror/mturk/idm_data/split/idm_2022-12-08_12h53.16.781/cdd/idm_2022-12-08_12h53.16.781_cdd.csv'
	CDD_fn = request_input_path(prompt='Please enter the path to an arbitray CDD file')
	df = pd.read_csv(CDD_fn)
	print(df)
	print(list(df))

	# indifference point for each possible set of task values, set SV_now = SV_later to find kappa
	df['kappa'] = ( df['cdd_delay_amt']/df['cdd_immed_amt'] - 1.0 ) / df['cdd_delay_wait']
	df['log_kappa'] = np.log(df['kappa'])
	df = df.sort_values(by=['kappa'])
	df_kappa = df[['kappa','log_kappa','cdd_immed_amt', 'cdd_immed_wait', 'cdd_delay_amt', 'cdd_delay_wait']]
	print(df_kappa)

	save_df(df_kappa,prompt='Please enter the path where to write kappa estimates')


if __name__ == "__main__":
	# main will be executed after running the script
    main()
