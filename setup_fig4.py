import numpy as np
from gens import sim_setup

dir_name='./setting4'

# Set seed for reproducibility
rseed = 555

n = 2000
d_Vec = [5000, 7500, 10000, 15000, 20000]
M_Vec = [10, 20, 30, 40, 50]
alpha = 0
Sigma_type = 'Gaus_ind'
K_Vec = [1, 2, 3]  # sparsity level
OMPFlag = True
DebLassoFlag = False
SISFlag = False
PrepmvXtFlag = False
prllz=1 #no parallelization

# SNR
noise_sigma = 1
t_min_Vec = [0.06]

# Number of iterations
NumOfSig = 500  # number of independent noise realizations

L_factor = 2

# Theta setup
theta_ind_type = 'first'
theta_sign_type = 'pos'
theta_amp_type = 'eq'

sim_setup(rseed, n, d_Vec, M_Vec, alpha, Sigma_type, K_Vec, OMPFlag, DebLassoFlag, SISFlag, PrepmvXtFlag, noise_sigma,
          t_min_Vec, NumOfSig, L_factor, dir_name, theta_ind_type, theta_sign_type, theta_amp_type, prllz=prllz)