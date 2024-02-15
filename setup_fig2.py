import numpy as np
from gens import sim_setup

dir_name='./setting2'

# Set seed for reproducibility
rseed = 555

n = 2000
d_Vec = [2500, 5000, 7500, 10000]
M_Vec = [20]
alpha = 0
Sigma_type = 'Gaus_ind'
K_Vec = [5]  # sparsity level
OMPFlag = True
DebLassoFlag = True
SISFlag = True
PrepmvXtFlag = False
prllz=1 #no parallelization - want to check actual running time

# SNR
noise_sigma = 1
t_min_Vec = [0.1]

# Number of iterations
NumOfSig = 20  # number of independent noise realizations

L_factor = 2

# Theta setup
theta_ind_type = 'first'
theta_sign_type = 'alt'
theta_amp_type = 'ggeq'

sim_setup(rseed, n, d_Vec, M_Vec, alpha, Sigma_type, K_Vec, OMPFlag, DebLassoFlag, SISFlag, PrepmvXtFlag, noise_sigma,
          t_min_Vec, NumOfSig, L_factor, dir_name, theta_ind_type, theta_sign_type, theta_amp_type, prllz=prllz)