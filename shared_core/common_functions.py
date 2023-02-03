import os,sys

def make_dir(this_dir,verbose=False):
    if not os.path.exists(this_dir):
        if verbose:
            print('Creating: {}'.format(this_dir))
        os.makedirs(this_dir)

def save_df(df,prompt='Please enter path to save df'):
    # e.g. '/Users/pizarror/mturk/idm_data/kappa_values.csv'
    fn = input(''.join([prompt,':\n']))
    make_dir(os.path.dirname(fn))
    print('Saving dataframe to : {}'.format(fn))
    df.to_csv(fn)


def request_input_path(prompt='Please enter the path to the file'):
    fn = input(''.join([prompt,':\n']))
    if not os.path.exists(fn):
        print('Could not find this path, please try again')
        sys.exit()
    else:
        return fn

def request_save_path(prompt='Please enter the path to where we will save the data'):
    fn = input(''.join([prompt,':\n']))
    make_dir(os.path.dirname(fn))
    return fn


def main(args):
    make_dir(args,verbose=True)

if __name__ == '__main__':
    main(sys.argv)



