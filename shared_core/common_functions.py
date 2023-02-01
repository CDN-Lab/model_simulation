import os,sys

def make_dir(this_dir,verbose=False):
    if not os.path.exists(this_dir):
        if verbose:
            print('Creating: {}'.format(this_dir))
        os.makedirs(this_dir)

def request_path(prompt='Please enter the path to the file'):
    fn = input(prompt + ':\n')
    if not os.path.exists(fn):
        print('Could not find this path, please try again')
        sys.exit()
    else:
        return fn


def main(args):
    make_dir(args,verbose=True)

if __name__ == '__main__':
    main(sys.argv)



