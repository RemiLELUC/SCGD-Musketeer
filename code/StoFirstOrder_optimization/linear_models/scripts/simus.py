import numpy as np
from tqdm import tqdm
from models import linearReg,logisticReg

def run_ols(X,y,𝜆,method,N_exp,N,batch_size,
            T,a,alpha_power,eta,unbiased,verbose):
    n_features = X.shape[1]
    w_res = np.zeros((N_exp,N+1,n_features))
    w0 = np.zeros(n_features)
    np.random.seed(0)
    loss_list = np.zeros((N_exp,N+1))
    counts_list = np.zeros((N_exp,n_features))
    model_ols = linearReg(X=X,y=y,𝜆=𝜆)
    for i in tqdm(range(N_exp)):
        counts,w_res_list = model_ols.fit(seed=i,x0=w0,N=N,batch_size=batch_size,
                                   T=T,a=a,alpha_power=alpha_power,eta=eta,
                                   method=method,unbiased=unbiased,
                                   verbose=verbose)
        w_res[i] = w_res_list
        counts_list[i] = counts
        loss = [model_ols.loss(w) for w in w_res_list]
        loss_list[i] = loss
    return w_res,counts_list,loss_list

def run_log(X,y,𝜆,fit_intercept,method,N_exp,N,batch_size,
            T,a,alpha_power,eta,unbiased,verbose):
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
        counts,w_res_list = model_log.fit(seed=i,x0=w0,N=N,batch_size=batch_size,
                                   T=T,a=a,alpha_power=alpha_power,eta=eta,
                                   method=method,unbiased=unbiased,
                                   verbose=verbose)
        counts_list[i] = counts
        loss = [model_log.loss(w) for w in w_res_list]
        w_res[i] = w_res_list
        loss_list[i] = loss
    return w_res,counts_list,loss_list
