import sys
import numpy as np
import algs
from algs import DOMP_sepX, deblasso_algs_sepX, var_screen_sepX
from gens import mvxt_gen_all
import pickle
import random
import gc
import warnings
from helpers import pic_load, pic_save
import os.path

if __name__ == "__main__":
    setting_num = sys.argv[1] # 1a / 1b / 2 / 3 / 4 - corrsponds to figures
    sig_Ind = int(sys.argv[2])-1 #sys.argv[2] = task_id

    data_folder = './setting' + setting_num + '/'
    print(data_folder)
    if not os.path.exists(data_folder+'results/'):
        os.mkdir(data_folder+'results/')

    # cleans memory
    gc.collect()

    #load parameters
    pars_saved = pic_load(data_folder + 'pars.dat')
    for key, val in pars_saved.items():
        exec(key + '=val')


    OMP_Num = len(OMP_alg_names)
    Lasso_Num = len(Lasso_alg_names)
    SIS_Num = len(SIS_alg_names)

    print("pars loaded")

    for t_min_Ind, t_min in enumerate(t_min_Vec):
        print("theta_min %d/%d" % (t_min_Ind+1, len(t_min_Vec)))
        for d_Ind, d in enumerate(d_Vec):
            print("d %d/%d" % (d_Ind + 1, len(d_Vec)))
            for M_Ind, M in enumerate(M_Vec):
                print("M %d/%d" % (M_Ind + 1, len(M_Vec)))
                for K_Ind, K in enumerate(K_Vec):
                    print("K %d/%d" % (K_Ind + 1, len(K_Vec)))
                    L = L_factor*K
                    result_filename = (data_folder + 'results/sig_Ind%d_t_min_Ind%d_d_Ind%d_M_Ind%d_K_Ind%d.pickle' % (
                    sig_Ind, t_min_Ind, d_Ind, M_Ind, K_Ind))
                    if os.path.exists(result_filename):
                        warnings.warn(result_filename+" already exists")
                    else:
                        print(result_filename + " in progress")
                        X_dir_name = data_folder +'/X_d_Ind%d/' %(d_Ind)
                        mumax = np.load(X_dir_name + 'mumax'+str(M)+'.npy')
                        mu_stacked = np.load(X_dir_name + 'mu_stacked'+str(M)+'.npy')
                        if DebLassoFlag:
                            mvXt_dir_name = data_folder +'/mvXt_d_Ind%d/' %(d_Ind)

                        # Does max-MIP hold?
                        if mumax > 1 / (2 * K - 1):
                            warnings.warn('max-MIP does not hold')

                        # Setting up theta and Y
                        true_inds = ind_mat[d_Ind,K_Ind][:, t_min_Ind]
                        theta = np.zeros(d)
                        theta[true_inds] = vals_mat[d_Ind,K_Ind][:, t_min_Ind]

                        Y_clean = np.zeros((n, M))
                        for m in range(M):
                            Xm = np.load(X_dir_name + str(m) + '.npy')
                            Y_clean[:, m] = np.matmul(Xm, theta)

                        # Set noise seed as sig_Ind for reproducibility
                        np.random.seed(sig_Ind)
                        noise = np.random.randn(n, M)
                        Y = Y_clean + noise_sigma * noise

                        Succ_sig = np.zeros(len(alg_names))
                        Duration_sig = np.zeros(len(alg_names))

                        if OMPFlag:
                            for algInd in range(len(OMP_alg_names)):
                                if OMP_alg_names[algInd].endswith('-L'):
                                    Succ_sig[algInd], Duration_sig[algInd] = DOMP_sepX(X_dir_name, Y, M, K, OMP_alg_names[algInd], true_inds, L)
                                else:
                                    Succ_sig[algInd], Duration_sig[algInd] = DOMP_sepX(X_dir_name, Y, M, K, OMP_alg_names[algInd], true_inds)

                        if DebLassoFlag:
                            algInd=OMP_Num
                            if PrepmvXtFlag==False: #If matrices were not generated in preprocessing, then generate them now
                                deb_lasso_setup_time = mvxt_gen_all(X_dir_name, mvXt_dir_name, d_Ind, d, n, M, noise_sigma, PrepmvXtFlag, mod_lambda)
                            sd_thetahat = np.load(mvXt_dir_name + 'sd_thetahat.npy')
                            Succ_sig[algInd], Succ_sig[algInd+1], deb_lasso_runtime = deblasso_algs_sepX(X_dir_name, Y, mvXt_dir_name, n, d, M, K, mod_lambda, sd_thetahat, true_inds)
                            Duration_sig[algInd] = deb_lasso_setup_time+deb_lasso_runtime

                        if SISFlag:
                            algInd = OMP_Num+Lasso_Num
                            Succ_sig[-len(SIS_alg_names):], Duration_sig[-len(SIS_alg_names):], = var_screen_sepX(X_dir_name, Y, M, K, SIS_alg_names, true_inds)


                        result = {}
                        result['Succ_sig'] = Succ_sig
                        result['Duration_sig'] = Duration_sig

                        # Save the result using pickle

                        with open(result_filename, 'wb') as f:
                            pickle.dump(result, f)
                        print(result_filename+" saved")
