import numpy as np
from gens import sim_setup

dir_name='./setting1a'

# Set seed for reproducibility
rseed = 555

n = 2000
d_Vec = [10000]
M_Vec = [20]
alpha = 0
Sigma_type = 'Gaus_ind'
K_Vec = [5]  # sparsity level
OMPFlag = True
DebLassoFlag = True
SISFlag = True
PrepmvXtFlag = True
prllz=40 # preprocessing parallel parameter

# SNR
noise_sigma = 1
t_min_Vec = np.concatenate((np.arange(0.25, 5, 0.1),np.arange(5, 7.5, 0.5))) / np.sqrt(n)

# Number of iterations
NumOfSig = 500  # number of independent noise realizations

L_factor = 2

# Theta setup
theta_ind_type = 'first'
theta_sign_type = 'alt'
theta_amp_type = 'ggeq'

sim_setup(rseed, n, d_Vec, M_Vec, alpha, Sigma_type, K_Vec, OMPFlag, DebLassoFlag, SISFlag, PrepmvXtFlag, noise_sigma,
          t_min_Vec, NumOfSig, L_factor, dir_name, theta_ind_type, theta_sign_type, theta_amp_type, prllz=prllz)