import numpy as np
from joblib import Parallel, delayed
from sklearn import linear_model
from scipy.linalg import toeplitz
import os
import time
import warnings
import pickle
from helpers import pic_save, pic_load
from numpy.linalg import pinv
from joblib import Parallel, delayed
from sklearn import linear_model
from timeit import default_timer as timer


def X_sep_gen(n, d, M, Sigma_type, alpha, save_folder_path, A=None):
    """Generate M separate X matrices according to specifications

    :param n: number of samples
    :param d: dimension of samples
    :param M: number of matrices
    :param Sigma_type: type of matrix (Gaus_ind / Gaus_corr / Gaus_corr_first_A_cols)
    :param alpha: correlation strength parameter
    :param save_folder_path: save location
    :param A: number of correlated columns
    :return: None
    The function saves following files in save_folder_path:
    M matrices numbered 0,...,M-1;
    mumax;
    mu_stacked - coherence of a (nM x d) matrix composed of stacking the M matrices
    """

    X = np.zeros((n, d, M))
    mu = np.zeros(M)

    if Sigma_type == 'Gaus_ind':
        # each matrix entry is i.i.d. standard Gaussian
        for m in range(M):
            print("Generating design matrix %d/%d" % (m+1,M))
            filename = save_folder_path + str(m) + '.npy'
            if os.path.exists(filename):
                X_unnorm = np.load(filename)
            else:
                X_unnorm = np.random.randn(n, d)
                np.save(filename, X_unnorm.astype(np.float32))
            X_norm = X_unnorm / np.linalg.norm(X_unnorm, axis=0)
            G = np.dot(X_norm.T, X_norm)
            mu[m] = np.max(np.abs(G - np.eye(d)))
            X[:, :, m] = X_unnorm
    elif Sigma_type == 'Gaus_corr':
        # Correlation between columns i and j is alpha^(i-j)
        r = alpha ** np.arange(d)
        Sigma = toeplitz(r)
        C = np.linalg.cholesky(Sigma)
        for m in range(M):
            print("Generating design matrix %d/%d" % (m+1,M))
            filename = save_folder_path + str(m) + '.npy'
            if os.path.exists(filename):
                X_unnorm = np.load(filename)
            else:
                X_ind = np.random.randn(n, d)
                X_unnorm = np.dot(X_ind, C.T)
                np.save(filename, X_unnorm.astype(np.float32))
            X_norm = X_unnorm / np.linalg.norm(X_unnorm, axis=0)
            G = np.dot(X_norm.T, X_norm)
            mu[m] = np.max(np.abs(G - np.eye(d)))
            X[:, :, m] = X_unnorm
    elif Sigma_type == 'Gaus_corr_first_A_cols':
        #first A columns have correlation alpha with every other column, the rest are independet of each other
        r=alpha*np.ones(d)
        Sigma = np.eye(d)
        for k in range(A):
            Sigma[k, :] = r
            Sigma[:, k] = r
        np.fill_diagonal(Sigma,1)
        C = np.linalg.cholesky(Sigma)
        for m in range(M):
            print("Generating design matrix %d/%d" % (m+1,M))
            filename = save_folder_path + str(m) + '.npy'
            if os.path.exists(filename):
                X_unnorm = np.load(filename)
            else:
                X_ind = np.random.randn(n, d)
                X_unnorm = np.dot(X_ind, C.T)
                np.save(filename, X_unnorm.astype(np.float32))
            X_norm = X_unnorm / np.linalg.norm(X_unnorm, axis=0)
            G = np.dot(X_norm.T, X_norm)
            mu[m] = np.max(np.abs(G - np.eye(d)))
            X[:, :, m] = X_unnorm

    else:
        warnings.warn("unrecognized X type")

    mumax = np.max(mu)

    X_stacked = np.reshape(np.transpose(X[:, :, :M], (2,0,1)), (n * M, d))
    X_stacked_norm = X_stacked / np.linalg.norm(X_stacked, axis=0)
    G = np.dot(X_stacked_norm.T, X_stacked_norm)
    mu_stacked = np.max(np.abs(G - np.eye(d)))

    np.save(save_folder_path+'mumax'+str(M)+'.npy', mumax)
    np.save(save_folder_path+'mu_stacked'+str(M)+'.npy', mu_stacked)


def theta_gen(d, K, t_min, ind_type, sign_type, amp_type):
    """
    :param d: dimension of theta
    :param K: sparsity level
    :param t_min: minimum non-zero value
    :param ind_type: where the non-zeros are located (rand / first / first_gaps)
    :param sign_type: structure of signs (pos / rand / alt)
    :param amp_type: relation between amplitudes of non-zero values (eq / geq / ggeq)
    :return: vals and ind of non-zero values of theta
    """

    ind_mapping = {
        'rand': np.random.choice(d, K, replace=False),
        'first': np.arange(K),
        'first_gaps': np.arange(0, 2 * K, 2)
    }
    ind = ind_mapping.get(ind_type, None)
    if ind is None:
        raise ValueError('Unrecognized ind_type')

    sign_mapping = {
        'pos': np.ones(K),
        'rand': np.sign(np.random.randn(K)),
        'alt': (-1) ** np.arange(K)
    }
    signs = sign_mapping.get(sign_type, None)
    if signs is None:
        raise ValueError('Unrecognized sign_type')

    amp_mapping = {
        'eq': t_min * np.ones(K),
        'geq': np.linspace(t_min, 2 * t_min, K),
        'ggeq': np.linspace(t_min, 3 * t_min, K)
    }
    coef = amp_mapping.get(amp_type, None)
    if coef is None:
        raise ValueError('Unrecognized amp_type')

    vals = signs * coef

    return vals, ind

def mvxt_gen(x, mod_lambda, prllz):
    """ Calculates a decorrelation term using the matrix x by calcuating a decorrelation matrix and multiplying it by
    the transpose of x. To save time, there is an option to parallelize this computation.
    :param x: design matrix
    :param mod_lambda: Lasso penalty parameter lambda
    :param prllz - parallelization paramter (default - no parallelization) - used to speedup simulation setup
    :return: product of decorrelation matrix mv with matrix x.T
    """

    n = x.shape[0]
    p = x.shape[1]

    m_c = np.ones((p, p))
    va = np.zeros(p)
    mod = linear_model.Lasso(alpha=mod_lambda, fit_intercept=False)

    # function to fit lasso to column jj of x in function of other columns
    def node_fitlasso(jj):
        x_til = np.reshape(x[:, jj], (n, 1))
        mod.fit(np.delete(x, jj, 1), x_til)
        aux = np.reshape(mod.coef_, (p - 1, 1))
        m_c_aux = np.insert(-aux, jj, 1)
        va_aux = (1 / n) * np.dot((x_til - np.matmul(np.delete(x, jj, 1), aux)).T, x_til)
        return np.array([va_aux, m_c_aux], dtype=object)

    results = Parallel(n_jobs=prllz)(delayed(node_fitlasso)(jj) for jj in range(p))
    for jj in range(p):
        va[jj] = results[jj][0]
        m_c[:][jj] = results[jj][1]

    mv = np.dot(np.diag(np.power(va, -1)), m_c)
    mvxt = np.dot(mv, x.T)

    return mvxt


def mvxt_gen_all(X_dir_name, mvXt_dirname, d_Ind, d, n, M_max, noise_sigma, PrepmvXtFlag, mod_lambda, prllz=1):
    start = timer()
    sd_thetahat = np.zeros((M_max, d))
    for m in range(M_max):
        mvXt_m_filename = mvXt_dirname + str(m) + '.npy'
        if PrepmvXtFlag and os.path.exists(mvXt_m_filename):
            mvXt_m = np.load(mvXt_m_filename)  # if mvXt is not currently loaded, then load it
        else:
            X_m_filename = X_dir_name + str(m) + '.npy'
            X_m = np.load(X_m_filename)
            # precomputing decorrelation matrix mv and mv*X.T, so we can avoid repeating this product in the simulation
            mvXt_m = mvxt_gen(X_m, mod_lambda, prllz)
            np.save(mvXt_m_filename, mvXt_m)
        for i in range(d):
            sd_thetahat[m, i] = noise_sigma * np.sqrt(np.sum(np.power(mvXt_m[i, :], 2)) / n) #standard deviation of debiaed lasso estimator
        print("Generating debiasing term %d/%d" % (m + 1, M_max))
    np.save(mvXt_dirname+'sd_thetahat.npy', sd_thetahat)
    end = timer()
    deb_lasso_setup_time = end - start #note that this is the time for the parallel computation
    return deb_lasso_setup_time






def sim_setup(rseed, n, d_Vec, M_Vec, alpha, Sigma_type, K_Vec, OMPFlag, DebLassoFlag, SISFlag, PrepmvXtFlag, noise_sigma,
              t_min_Vec, NumOfSig, L_factor, dir_name, theta_ind_type, theta_sign_type, theta_amp_type, prllz=1, DJOMPunFlag=False):
    """ Sets up simulation according to specified parameters

    :param rseed: random seed (for reproducibility)
    :param n: number of samples per machine
    :param d_Vec: vector of dimensions of theta
    :param M_Vec: vector of number of machines
    :param alpha: correlation strength parameter
    :param Sigma_type: type of matrix (Gaus_ind / Gaus_corr / Gaus_corr_first_A_cols)
    :param K_Vec: vector of sparsity levels
    :param OMPFlag: binary, execute OMP algorithms?
    :param DebLassoFlag: binary, execute Debiaed Lasso algorithms?
    :param SISFlag: binary, execute SIS algorithms?
    :param noise_sigma: noise level
    :param t_min_Vec: vector of theta_min values
    :param NumOfSig: Number of repetitions / independet realizations of noise
    :param L_factor: L=L_factor*K is parameter of D-OMP algorithm
    :param dir_name: save location
    :param theta_ind_type: where the non-zeros of theta are located (rand / first / first_gaps)
    :param theta_sign_type: structure of signs (pos / rand / alt)
    :param theta_amp_type: relation between amplitudes of non-zero values of theta (eq / geq / ggeq)
    :param prllz: parallelization paramter (default - no parallelization) - used to speedup simulation setup
    :return: None
    The function creates a pickle file named pars, calls functions that generates for each value d of d_Vec and value M
    of M_Vec, M design matrices of size n-by-d and a sparse vector theta of size d. If DebLassoFlag==True, it also
    generates M decorrelation matrices and computes their product with their corresponding transposed design matrices.
    All of the above are saved at dir_name.
    """

    np.random.seed(rseed)
    NumOft_min = len(t_min_Vec)
    NumOfd = len(d_Vec)
    NumOfK = len(K_Vec)
    NumOfM = len(M_Vec)
    M_max = max(M_Vec)

    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)

    # Algorithms for comparison:
    if DJOMPunFlag:
        OMP_alg_names = ['DJ-OMP', 'DJ-OMP-un']
    else:
        OMP_alg_names = ['OMP-one', 'DJ-OMP', 'D-OMP-K', 'D-OMP-L']
    Lasso_alg_names = ['AvgDebLasso', 'BNM21-K']
    SIS_alg_names = [ 'SIS-OMP-py']
    alg_names = OMP_alg_names + Lasso_alg_names + SIS_alg_names

    ind_mat = {}
    vals_mat = {}
    for d_Ind, d in enumerate(d_Vec):
        print("d = %d" % (d))

        for K_Ind, K in enumerate(K_Vec):
            print("K = %d" % (K))
            ind_mat[d_Ind,K_Ind] = np.zeros((K, NumOft_min))
            vals_mat[d_Ind,K_Ind] = np.zeros((K, NumOft_min))

            # generate theta vector for each value of t_min
            for t_min_Ind in range(NumOft_min):
                t_min = t_min_Vec[t_min_Ind]
                vals_mat[d_Ind,K_Ind][:, t_min_Ind], ind_mat[d_Ind,K_Ind][:, t_min_Ind] = theta_gen(d, K, t_min, theta_ind_type,
                                                                                        theta_sign_type, theta_amp_type)

            ind_mat[d_Ind,K_Ind] = ind_mat[d_Ind,K_Ind].astype(int)



        X_dir_name = './X_d_Ind%d/' %(d_Ind)

        if not os.path.exists(X_dir_name):
            os.mkdir(X_dir_name)
        for M_Ind, M in enumerate(M_Vec):
            # generate input matrices X^m
            X_sep_gen(n, d, M, Sigma_type, alpha, X_dir_name)

        if DebLassoFlag:
            mvXt_dirname = './mvXt_d_Ind%d/' % (d_Ind)
            if not os.path.exists(mvXt_dirname):
                os.mkdir(mvXt_dirname)
            # fixed penalty for the lasso
            mod_lambda = 2 * noise_sigma * np.sqrt(np.log(d / n))
            if PrepmvXtFlag:
                deb_lasso_setup_time = mvxt_gen_all(X_dir_name, mvXt_dirname, d_Ind, d, n, M_max, noise_sigma, PrepmvXtFlag, mod_lambda, prllz)



    if DebLassoFlag:
        if PrepmvXtFlag:
            pic_save('pars.dat', rseed=rseed, n=n, d_Vec=d_Vec, M_Vec=M_Vec, alpha=alpha, K_Vec=K_Vec, noise_sigma=noise_sigma,
                     t_min_Vec=t_min_Vec, NumOft_min=NumOft_min, NumOfSig=NumOfSig, L_factor=L_factor, ind_mat=ind_mat,
                     vals_mat=vals_mat,
                     DebLassoFlag=DebLassoFlag, SISFlag=SISFlag, OMPFlag=OMPFlag, PrepmvXtFlag=PrepmvXtFlag, Sigma_type=Sigma_type,
                     OMP_alg_names=OMP_alg_names, Lasso_alg_names=Lasso_alg_names, SIS_alg_names=SIS_alg_names, alg_names=alg_names,
                     mod_lambda=mod_lambda, deb_lasso_setup_time=deb_lasso_setup_time)
        else:
           pic_save('pars.dat', rseed=rseed, n=n, d_Vec=d_Vec, M_Vec=M_Vec, alpha=alpha, K_Vec=K_Vec, noise_sigma=noise_sigma,
                 t_min_Vec=t_min_Vec, NumOft_min=NumOft_min, NumOfSig=NumOfSig, L_factor=L_factor, ind_mat=ind_mat, vals_mat=vals_mat,
                 DebLassoFlag=DebLassoFlag, SISFlag=SISFlag, OMPFlag=OMPFlag, PrepmvXtFlag=PrepmvXtFlag, Sigma_type=Sigma_type,
                 OMP_alg_names=OMP_alg_names, Lasso_alg_names=Lasso_alg_names, SIS_alg_names=SIS_alg_names, alg_names=alg_names,
                 mod_lambda=mod_lambda)
    else:
        pic_save('pars.dat', rseed=rseed, n=n, d_Vec=d_Vec, M_Vec=M_Vec, alpha=alpha, K_Vec=K_Vec, noise_sigma=noise_sigma,
                 t_min_Vec=t_min_Vec, NumOft_min=NumOft_min, NumOfSig=NumOfSig, L_factor=L_factor, ind_mat=ind_mat, vals_mat=vals_mat,
                 DebLassoFlag=DebLassoFlag, SISFlag=SISFlag, OMPFlag=OMPFlag, PrepmvXtFlag=PrepmvXtFlag, Sigma_type=Sigma_type,
                 OMP_alg_names=OMP_alg_names, Lasso_alg_names=Lasso_alg_names, SIS_alg_names=SIS_alg_names, alg_names=alg_names)
    print("Setup completed")