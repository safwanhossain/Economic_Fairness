# This file contains default values for global parameters
ns = 100		# number of training samples			set to 500
nt = 150                # number of test samples
m = 16			# length of feature vector x - set to 32
d = 5			# number of classes
num_groups = 4	    # black and white
lambda_welf = 2.2     # welfare is not really a constraint - so leave a small lambda
lambda_equi = 1.0   # for equitability, the constraint is well defined. So make it very high
dist = 'uniform'    # distribution defined options: 'normal' and 'uniform'
eps = 0.1           # the error margin for
hard_cons_eps = 0.2
num_sims = 20
NUM_CORES=16
K=4
