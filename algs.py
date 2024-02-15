import numpy as np
from numpy.linalg import pinv
from sklearn import linear_model
from timeit import default_timer as timer


# import package that loads R packages, for comparison with SIS:
import rpy2
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

# Initialize an R instance
r = robjects.r

# imports modules for R
base = importr("base")
utils = importr("utils")
# Activate automatic conversion for numpy arrays
numpy2ri.activate()
# Import R package
SIS = importr('SIS')



def center_maj_voting(votes, values, K):
    """ outputs the K indices with most votes

    :param votes: a vector consisting of indices sent by different machines
    :param values: a vector consisting of values that correspond to the
    indices in "votes", truncated to precision of O(log(d)).
    This is an optional argument that is used for tie-breaking.
    :param K: number of indices to be selected
    :return: Inds_of_top_K_Votes - a vector of K indices that received the maximal
    number of votes. In case of ties, the index with maximal sum of "values" is selected.
    If "values" is empty, then ties are broken arbitrarily.
    """

    votes = np.array(votes)
    values = np.array(values)

    # Create matrix of indices and the sum of votes that they received
    # and then sort indices by ascending number of votes
    C, ic = np.unique(votes, return_inverse=True)
    a_counts = np.bincount(ic)
    Num_of_Machines_choosing_Ind = np.column_stack((C, a_counts))

    max_num_of_votes = 0
    if Num_of_Machines_choosing_Ind.shape[0] < K:
        # not sufficient votes for exact support recovery -- failure
        Inds_of_top_K_Votes = np.array([])
    else:
        Inds_of_Votes_Sorted = Num_of_Machines_choosing_Ind[np.argsort(Num_of_Machines_choosing_Ind[:, 1])]
        # Add to estimated support set indices with maximal number of votes
        # one by one, starting with those with the most votes. In case of ties,
        # the values are used as tie-breakers
        Inds_of_top_K_Votes = []
        while len(Inds_of_top_K_Votes) < K:
            max_num_of_votes = Inds_of_Votes_Sorted[-1, 1]
            num_of_machines_with_max_votes = np.sum(Inds_of_Votes_Sorted[:, 1] == max_num_of_votes)
            if num_of_machines_with_max_votes <= K - len(Inds_of_top_K_Votes):
                # no ties
                Inds_of_top_K_Votes.extend(Inds_of_Votes_Sorted[Inds_of_Votes_Sorted[:, 1] == max_num_of_votes, 0])
                Inds_of_Votes_Sorted = Inds_of_Votes_Sorted[:-num_of_machines_with_max_votes, :]
            elif values.size == 0:
                # break ties at random
                cands = Inds_of_Votes_Sorted[-num_of_machines_with_max_votes:, 0]
                rand_cands_with_max_votes = np.random.choice(cands, size=K - len(Inds_of_top_K_Votes), replace=False)
                Inds_of_top_K_Votes.extend(rand_cands_with_max_votes)
            else:
                # use values to break ties
                cands = np.zeros((num_of_machines_with_max_votes, 2))
                cands[:, 0] = Inds_of_Votes_Sorted[-num_of_machines_with_max_votes:, 0]  # indexes
                for j in range(len(cands[:, 0])):
                    cands[j, 1] = np.sum(values[votes == cands[j, 0]])  # scores
                sorted_cands = cands[np.argsort(cands[:, 1])]
                Inds_of_top_K_Votes.extend(sorted_cands[-(K - len(Inds_of_top_K_Votes)):, 0])

        Inds_of_top_K_Votes = np.sort(Inds_of_top_K_Votes)

    return Inds_of_top_K_Votes, max_num_of_votes


def DOMP_sepX(X_dir_name, Y, M, K, alg_name, true_inds, L=None):
    """ Distributed-OMP algorithms - DJ-OMP, D-OMP, and OMP using only one machine

    :param X_dir_name: design matrices location
    :param Y: matrix of response vectors
    :param M: number of machines
    :param K: sparsity level
    :param alg_name: specified algorithm ('DJ-OMP'/'D-OMP-K'/'D-OMP-L'/'OMP-one')
    :param true_inds: underlying support set
    :param L: number of steps for D-OMP algorithm
    :return:
    Succ_sig (binary) - 1 only if estimated support set is identical to true_inds;
    duration (float) - execution time duration
    """

    if L is None:
        L=K

    estimated_support = []

    start = timer()
    if alg_name == 'DJ-OMP': # DJ-OMP
        votes = np.zeros(M)
        values = np.zeros(M)
        for k in range(K):
            for m in range(M):
                Xm = np.load(X_dir_name + str(m) + '.npy')
                votes[m], values[m] = OMP(Xm, Y[:, m], estimated_support, 1)
            Ind_of_top_Vote, _ = center_maj_voting(votes, [], 1)
            estimated_support.append(int(Ind_of_top_Vote))

    elif alg_name == 'DJ-OMP-un': # DJ-OMP with unknown K
        votes = np.zeros(M)
        values = np.zeros(M)
        stopping_crit=False
        while not stopping_crit:
            for m in range(M):
                Xm = np.load(X_dir_name + str(m) + '.npy')
                votes[m], values[m] = OMP(Xm, Y[:, m], estimated_support, 1)
            Ind_of_top_Vote, max_num_of_votes = center_maj_voting(votes, [], 1)
            if max_num_of_votes<2:
                stopping_crit = True
            else:
                estimated_support.append(int(Ind_of_top_Vote))

    elif alg_name == 'D-OMP-K': # D-OMP with L=K
        votes_mat = np.zeros((M, K))
        values_mat = np.zeros((M, K))
        for m in range(M):
            Xm = np.load(X_dir_name + str(m) + '.npy')
            votes_mat[m, :], values_mat[m, :] = OMP(Xm, Y[:, m], [], K)
        estimated_support, _ = center_maj_voting(votes_mat, [], K)

    elif alg_name == 'D-OMP-L': # D-OMP with specified L
        votes_mat = np.zeros((M, L))
        values_mat = np.zeros((M, L))
        for m in range(M):
            Xm = np.load(X_dir_name + str(m) + '.npy')
            votes_mat[m, :], values_mat[m, :] = OMP(Xm, Y[:, m], [], L)
        estimated_support, _ = center_maj_voting(votes_mat, [], K)

    elif alg_name == 'OMP-one':
        votes_mat = np.zeros((M, K))
        values_mat = np.zeros((M, K))
        for m in range(M):
            Xm = np.load(X_dir_name + str(m) + '.npy')
            votes_mat[m, :], values_mat[m, :] = OMP(Xm, Y[:, m], [], K)
        Succ_sig_machine = 0
        for m in range(M):
            if set(votes_mat[m, :]) == set(true_inds):
                Succ_sig_machine += 1
        Succ_sig = Succ_sig_machine / M  # percentage of successful machines

    else:
        raise ValueError('Unrecognized alg_name')

    if alg_name != 'OMP-one':
        if estimated_support is not None and set(estimated_support) == set(true_inds):   #np.sum(estimated_support != sorted_ind) == 0:  # precise support recovery
            Succ_sig = 1
        else:
            Succ_sig = 0
    end = timer()
    duration = end - start
    return Succ_sig, duration



def max_inner_prod(X_norm, r, N):
    """ N highest inner products between columns of X_norm and r

    :param X_norm: matrix with unit norm columns
    :param r: residual vector
    :param N: integer, number of inner products to return
    :return:
    ind - vector of indices of size N;
    val - vector of values of size N
    """

    inner_products = r.T @ X_norm  # Calculate inner products
    abs_inner_products = np.abs(inner_products)  # Calculate inner products
    ind = np.argsort(-abs_inner_products)[:N] # sort and take top-N indices
    val = inner_products[ind] # corresponding values
    return ind, val

def OMP(Xm, ym, estimated_support_in, K_out):
    """  Given initial support set estimated_support_in, performs K_out steps of Orthogonal Matching Pursuit.
    Note that this algorithm only involves one machine.

    :param Xm: design matrix on machine m
    :param ym: response vector on machine m
    :param estimated_support_in: initial estimated support set
    :param K_out: number of non-zero indices to be recovered
    :return:
    ind - vector of indices of new estimated support (size K_out);
    val - vector of values of absolute inner product of Xm and r of size K_out
    """

    n, d = Xm.shape

    # Initialization
    estimated_theta = np.zeros((d, ))
    if len(estimated_support_in): #not empty
        Xm_sliced = Xm[:, estimated_support_in]
        estimated_theta_on_support = pinv(Xm_sliced) @ ym
        estimated_theta[estimated_support_in] = estimated_theta_on_support
        r = ym - Xm @ estimated_theta  # residual updated to exclude part in direction on selected atom
    else:
        r = ym

    col_norms = np.linalg.norm(Xm, axis=0) ** 2  # \ell_2 norm ^2
    Xm_norm = Xm / col_norms

    K_in = len(estimated_support_in)
    estimated_support = np.concatenate((estimated_support_in, np.zeros((K_out, ))), axis=0)
    estimated_support = estimated_support.astype(int)
    estimated_theta = np.zeros((d, ))
    ind = np.zeros((K_out, )).astype(int)
    val = np.zeros((K_out, ))

    # In each step, find column that has maximal correlation with residual
    for itr in range(K_out):
        ind[itr], val[itr] = max_inner_prod(Xm_norm, r, 1)
        if itr < K_out - 1:
            estimated_support[K_in + itr] = ind[itr]  # Chosen atom added to support
            Xm_sliced = Xm[:, estimated_support[:K_in + itr + 1]]
            estimated_theta_on_support = pinv(Xm_sliced) @ ym
            estimated_theta[estimated_support[:K_in + itr + 1]] = estimated_theta_on_support
            r = ym - Xm @ estimated_theta  # Residual updated to exclude part in direction on selected atoms

    return ind, val


def deblasso_algs_sepX(X_dir_name, Y, mvXt_dir_name, n, d, M, K, mod_lambda, sd_thetahat, true_inds):
    """ Performs two versions of Debiased Lasso - (1) 'avgdeb' by Battey et al. (2018); and (2) 'bmn21_K', a variant of
    an algorithm proposed by Barghi et al (2021) whereby each machine sends the top-K indices of its debiased lasso
    estimation to the center, which performs majority voting.

    :param X_dir_name: design matrices location
    :param Y: matrix of response vectors
    :param mvXt_dir_name: preprocessed decorrelation terms location
    :param n: number of samples in each machine
    :param d: dimension of theta
    :param M: number of machines
    :param K: sparsity level
    :param mod_lambda: Lasso penalty parameter lambda
    :param sd_thetahat: standard deviation of debiaed lasso estimator
    :param true_inds: underlying support set
    :return:
    Succ_sig_avgdeb (binary) - 1 only if support set estimated by avgdeb is identical to true_inds;
    Succ_sig_bnm21_K (binary) - 1 only if support set estimated by bnm21_K is identical to true_inds;
    duration - execution time duration (the same for both algorithms)
    """

    # setting up the lasso
    mod = linear_model.Lasso(alpha=mod_lambda, fit_intercept=False)

    # votes from each machine for bnm21_K
    bnm21_K_votes = np.zeros(M, dtype=object)

    # average estimates of all local machines - will be the combined debiased lasso estimator
    theta_hat_avg = np.zeros(d)

    # loop for the data for each machine
    start = timer()
    for m in range(M):
        # fitting the lasso
        Xm = np.load(X_dir_name + str(m) + '.npy') #design matrix
        mvXt_m = np.load(mvXt_dir_name + str(m) + '.npy') #decorrelation matrix mv times the design matrix transpose
        mod.fit(Xm, Y[:, m])
        # lasso estimate
        theta_til = mod.coef_
        # debiased lasso estimate
        theta_hat = theta_til + (1 / n) * np.dot(mvXt_m, Y[:, m] - np.dot(Xm, theta_til))
        theta_hat_avg += theta_hat / M
        # standardized debiased lasso estimate
        vxi = np.sqrt(n) * np.divide(theta_hat, sd_thetahat[m, :])

        # BNM21: indices of top K values of vxi
        id_top = (np.argsort(abs(vxi))[-K:]).tolist()
        bnm21_K_votes[m] = id_top

    count_bnm21 = np.zeros(d)
    for m in range(M):
    # counting how many votes each index received
        for i in range(len(bnm21_K_votes[m])):
            count_bnm21[int(bnm21_K_votes[m][i])] += 1

    end = timer()
    duration = end - start
    # support estimate of each algorithm
    supphat_avgdeb = (np.argpartition([abs(value) for value in theta_hat_avg], -K)[-K:]).tolist()
    supphat_bnm21_K = (np.argpartition(count_bnm21, -K)[-K:]).tolist()

    # Results
    Succ_sig_avgdeb = set(supphat_avgdeb) == set(true_inds)
    Succ_sig_bnm21_K = set(supphat_bnm21_K) == set(true_inds)


    return Succ_sig_avgdeb, Succ_sig_bnm21_K, duration

def var_screen_sepX(X_dir_name, Y, M, K, SIS_alg_names, true_inds):
    """ distributed variable screening algorithms based on sure independence screening (SIS), implemention based on
    SIS R package.

    :param X_dir_name: design matrices location
    :param Y: matrix of response vectors
    :param M: number of machines
    :param K: sparsity level
    :param SIS_alg_names: which algorithms to execute (SIS-SCAD/SIS-OMP)
    :param true_inds: underlying support set
    :return:
    Succ_SIS_sig - 1 only if support set estimated by a given SIS-based distributed algorithm is identical to true_inds;
    duration - execution time duration for each SIS-based distributed algorithm
    """
    # [n, d, _] = X.shape

    votes_SIS = {}
    estimated_support_SIS = {}
    for method in SIS_alg_names:
        votes_SIS[method] = []
        estimated_support_SIS[method] = []
    Succ_SIS_sig = np.zeros(len(SIS_alg_names))
    num_inds_that_passed_screening_sig = np.zeros(len(SIS_alg_names))
    prcnt_of_true_inds_contained_sig = np.zeros(len(SIS_alg_names))
    full_support_contained_sig = np.zeros(len(SIS_alg_names))

    duration = np.zeros(len(SIS_alg_names))
    for m in range(M):
        Xm = np.load(X_dir_name + str(m) + '.npy')
        [n, d ] = Xm.shape
        Ym = Y[:, m]
        # Convert NumPy arrays to R matrices
        Xm_r = robjects.r['matrix'](Xm, nrow=n, ncol=d)
        Ym_r = robjects.r['matrix'](Ym, nrow=n, ncol=1)


        for method_ind, method in enumerate(SIS_alg_names):
            start = timer()
            if method == 'SIS-SCAD':
                model = SIS.SIS(x=Xm_r, y=Ym_r, family='gaussian', iter=False)
                # sis.ix0 are the variables that passed the screening, before running any additional algorithm
                # Include a correcting step - in R indices begin at 1, not 0
                inds_that_passed_screening = np.array(model.rx2('sis.ix0')) - 1
                votes = np.array(model.rx2('ix')) - 1
            elif method == 'SIS-OMP-R':
                model = SIS.SIS(x=Xm_r, y=Ym_r, family='gaussian', iter=False)
                inds_that_passed_screening = np.array(model.rx2('sis.ix0')) - 1
                Xm_screened = Xm[:,inds_that_passed_screening]
                inds_after_OMP, _ = OMP(Xm_screened, Ym, [], K)
                votes = inds_that_passed_screening[inds_after_OMP]
            elif method == 'SIS-OMP-py':
                # screening step - dimension reduction to n/log(n):
                d_tag = int(np.round(n/np.log(n)))
                col_norms = np.linalg.norm(Xm, axis=0) ** 2  # \ell_2 norm ^2
                Xm_norm = Xm / col_norms
                inds_that_passed_screening, _ = max_inner_prod(Xm_norm, Ym, d_tag)
                Xm_screened = Xm[:,inds_that_passed_screening]
                inds_after_OMP, _ = OMP(Xm_screened, Ym, [], K)
                votes = inds_that_passed_screening[inds_after_OMP]
            votes_SIS[method].append(votes)
            end = timer()
            duration[method_ind] += (end - start)

            num_inds_that_passed_screening_sig[method_ind] += len(inds_that_passed_screening)/M
            intersection = np.intersect1d(true_inds, inds_that_passed_screening)
            prcnt_of_true_inds_contained_sig[method_ind] += (len(intersection)/K)/M
            full_support_contained_sig[method_ind] += (set(intersection) == set(true_inds))/M


    for method_ind, method in enumerate(SIS_alg_names):
        votes_SIS[method] = np.concatenate(votes_SIS[method], axis=None)

        estimated_support_SIS[method], _ = center_maj_voting(votes_SIS[method], [], K)
        Succ_SIS_sig[method_ind] = set(estimated_support_SIS[method]) == set(true_inds)


    return Succ_SIS_sig, duration


