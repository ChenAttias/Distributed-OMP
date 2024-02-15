import os
import numpy as np
import warnings
import pickle

def pic_save(file_name, **nargs):
    # save many arguments at once using pickle
    import pickle, os
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            nargs = {**pickle.load(f), **nargs}
    with open(file_name, 'wb') as f:
        pickle.dump(nargs, f)


def pic_load(file_name, *pargs, **nargs):
    # load many arguments at once using pickle
    import pickle, os
    d = {}
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            d = pickle.load(f)
    if len(pargs) + len(nargs) == 0:
        return d
    r = (
        tuple(d[k] for k in pargs) +
        tuple(d.get(k, defv) for k, defv in nargs.items())
    )
    return r if len(pargs) + len(nargs) != 1 else r[0]


def aggregate_results(data_folder, NumOfd, NumOfM, NumOfK, NumOft_min, NumOfAlgs, NumOfSig):
    Succ_mat = np.zeros((NumOfd, NumOfM, NumOfK, NumOft_min, NumOfAlgs, NumOfSig))
    Duration_mat = np.zeros((NumOfd, NumOfM, NumOfK, NumOft_min, NumOfAlgs, NumOfSig))

    for d_Ind in range(NumOfd):
        for M_Ind in range(NumOfM):
            for K_Ind in range(NumOfK):
                for t_min_Ind in range(NumOft_min):
                    for sig_Ind in range(NumOfSig):
                        result_filename = (
                                    data_folder + 'results/sig_Ind%d_t_min_Ind%d_d_Ind%d_M_Ind%d_K_Ind%d.pickle' %
                                    (sig_Ind, t_min_Ind, d_Ind, M_Ind, K_Ind))
                        result = pic_load(result_filename)
                        if result:
                            Succ_mat[d_Ind,M_Ind,K_Ind,t_min_Ind,:,sig_Ind] += result['Succ_sig']
                            Duration_mat[d_Ind,M_Ind,K_Ind,t_min_Ind,:,sig_Ind] += result['Duration_sig']
                        else:
                            print("missing file " + result_filename)
    return Succ_mat, Duration_mat

def alg_names_for_display(alg_name):
    if alg_name == 'DJ-OMP':
        alg_name_for_display = r'$\mathtt{DJ-OMP}$'
    elif alg_name == 'D-OMP-K':
        alg_name_for_display = r'$\mathtt{D-OMP}$, $L=K$'
    elif alg_name == 'D-OMP-L':
        alg_name_for_display = r'$\mathtt{D-OMP}$, $L=2K$'
    elif alg_name == 'OMP-one':
        alg_name_for_display = r'Single $\mathtt{OMP}$'
    elif alg_name == 'AvgDebLasso':
        alg_name_for_display = r'$\mathtt{Deb-Lasso}$'
    elif alg_name == 'BNM21-K':
        alg_name_for_display = r'$\mathtt{Deb-Lasso-K}$'
    elif alg_name == 'SIS-SCAD':
        alg_name_for_display = r'$\mathtt{SIS-SCAD-K}$'
    elif alg_name == 'SIS-OMP':
        alg_name_for_display = r'$\mathtt{SIS-OMP-K}$'
    elif alg_name == 'SIS-OMP-py':
        alg_name_for_display = r'$\mathtt{SIS-OMP-K}$'
    elif alg_name == 'SIS-OMP-R':
        alg_name_for_display = r'$\mathtt{SIS-OMP-K}$ $\mathtt{(R)}$'
    elif alg_name == 'DJ-OMP-un':
        alg_name_for_display = r'$\mathtt{DJ-OMP}^{*}$'
    return alg_name_for_display


