import numpy as np
from tqdm import tqdm
from models import linearReg,logisticReg,my_svm

def run_ols(X,y,𝜆,batch_size,method,N_exp,N,
            T,gamma,mu_power,a,t0,alpha_power,
            fixed,eta,importance,gains,verbose):
    n_features = X.shape[1]
    w_res = np.zeros((N_exp,N+1,n_features))
    w0 = np.zeros(n_features)
    np.random.seed(0)
    loss_list = np.zeros((N_exp,N+1))
    counts_list = np.zeros((N_exp,n_features))
    model_ols = linearReg(X=X,y=y,𝜆=𝜆)
    for i in tqdm(range(N_exp)):
        counts,w_res_list = model_ols.fit(seed=41+i,x0=w0,N=N,batch_size=batch_size,
                                   T=T,gamma=gamma,mu_power=mu_power,
                                   a=a,t0=t0,alpha_power=alpha_power,fixed=fixed,
                                   eta=eta,method=method,
                                   importance=importance,gains=gains,
                                   verbose=verbose)
        w_res[i] = w_res_list
        counts_list[i] = counts
        loss = [model_ols.loss(np.arange(X.shape[0]),w) for w in w_res_list]
        loss_list[i] = loss
    return w_res,counts_list,loss_list

def run_log(X,y,𝜆,batch_size,fit_intercept,method,N_exp,N,
            T,gamma,mu_power,a,t0,alpha_power,fixed,
            eta,importance,gains,verbose):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    if fit_intercept:
        w0 = np.zeros(n_features+1)
        w_res = np.zeros((N_exp,N+1,n_features+1))
        counts_list = np.zeros((N_exp,n_features+1))
    else:
        w0 = np.zeros(n_features)
        w_res = np.zeros((N_exp,N+1,n_features))
        counts_list = np.zeros((N_exp,n_features))
    loss_list = np.zeros((N_exp,N+1))
    model_log = logisticReg(X=X,y=y,𝜆=𝜆,fit_intercept=fit_intercept)
    for i in tqdm(range(N_exp)):
        counts,w_res_list = model_log.fit(seed=42+i,x0=w0,N=N,batch_size=batch_size,
                                   T=T,gamma=gamma,mu_power=mu_power,
                                   a=a,t0=t0,alpha_power=alpha_power,fixed=fixed,eta=eta,
                                   method=method,
                                   importance=importance,gains=gains,
                                   verbose=verbose)
        counts_list[i] = counts
        loss = [model_log.loss(np.arange(X.shape[0]),w) for w in w_res_list]
        w_res[i] = w_res_list
        loss_list[i] = loss
    return w_res,counts_list,loss_list

def run_svm(X,y,𝜆,batch_size,method,N_exp,N,
            T,gamma,mu_power,a,t0,alpha_power,fixed,
            eta,importance,gains,verbose):
    n_features = X.shape[1]
    n_samples = X.shape[0]
    w0 = np.zeros(n_features)
    w_res = np.zeros((N_exp,N+1,n_features))
    counts_list = np.zeros((N_exp,n_features))
    loss_list = np.zeros((N_exp,N+1))
    model_svm = my_svm(X=X,y=y,𝜆=𝜆)
    for i in tqdm(range(N_exp)):
        counts,w_res_list = model_svm.fit(seed=42+i,x0=w0,N=N,batch_size=batch_size,
                                   T=T,gamma=gamma,mu_power=mu_power,
                                   a=a,t0=t0,alpha_power=alpha_power,fixed=fixed,eta=eta,
                                   method=method,
                                   importance=importance,gains=gains,
                                   verbose=verbose)
        counts_list[i] = counts
        loss = [model_svm.loss(np.arange(X.shape[0]),w) for w in w_res_list]
        w_res[i] = w_res_list
        loss_list[i] = loss
    return w_res,counts_list,loss_list
