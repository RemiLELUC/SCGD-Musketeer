import numpy as np
from utils import simu_block_log
from simus import run_log
from sklearn.linear_model import LogisticRegression

# Parameters
n_samples = 10000   # number of samples
n_features = 250    # dimension of the problem
𝜆 = 1/(n_samples)#regularization parameter
# Simulate data for regression
seed=0
puiss=2 
noise=0.01
block_size=1

X,y=simu_block_log(seed=seed,n_samples=n_samples,n_features=n_features,
                     puiss=puiss,block_size=block_size,noise=noise)

c = 1/(n_samples*λ)
log_sk = LogisticRegression(C=c,fit_intercept=False,tol=1e-3)
# fit sklearn model
log_sk.fit(X=X,y=y)
coeff = log_sk.coef_[0]

data_term  = np.log(1+np.exp(np.multiply(-y,X@coeff))).mean()
reg_term = (𝜆/2)*sum(coeff**2)
print('data_term:',data_term)
print('reg_term :',reg_term)
# Optimal loss
loss_opt = data_term + reg_term
print('loss_opt :',loss_opt)

N = int(1e3)    # number of passes over coordinates
a = 1         # numerator of learning rate
alpha_power = 1 # power in the learning rate
batch_size = 32   # batch size for gradient estimates
verbose = False # to display information
#T = int(n_features) # size of exploration
T = int(np.sqrt(n_features))
N_exp = 20     # number of experiments
print('T=',T)
print('𝜆=')
eta=1
print('eta=',eta)

# adaptive
_,_,loss_sgd = run_log(X=X,y=y,𝜆=𝜆,method='sgd',N_exp=N_exp,N=N,fit_intercept=False,
                       batch_size=batch_size,T=None,a=a,alpha_power=alpha_power,
                       eta=None,unbiased=None,verbose=False)
np.save('./results/LOGISTIC/loss_sgd_alpha{}.npy'.format(puiss),loss_sgd)

_,count_uni,loss_uni = run_log(X=X,y=y,𝜆=𝜆,method='uni',N_exp=N_exp,N=N,fit_intercept=False,
                                batch_size=batch_size,T=None,a=a,alpha_power=alpha_power,
                                eta=None,unbiased=None,verbose=False)
np.save('./results/LOGISTIC/loss_uni_alpha{}.npy'.format(puiss),loss_uni)
np.save('./results/LOGISTIC/count_uni_alpha{}.npy'.format(puiss),count_uni)

_,count_bia,loss_bia = run_log(X=X,y=y,𝜆=𝜆,method='musketeer',N_exp=N_exp,N=N,fit_intercept=False,
                                batch_size=batch_size,T=T,a=a,alpha_power=alpha_power,
                                eta=eta,unbiased=False,verbose=False)
np.save('./results/LOGISTIC/loss_bia_alpha{}.npy'.format(puiss),loss_bia)
np.save('./results/LOGISTIC/count_bia_alpha{}.npy'.format(puiss),count_bia)

_,count_unb,loss_unb= run_log(X=X,y=y,𝜆=𝜆,method='musketeer',N_exp=N_exp,N=N,fit_intercept=False,
                               batch_size=batch_size,T=T,a=a,alpha_power=alpha_power,
                               eta=eta,unbiased=True,verbose=False)
np.save('./results/LOGISTIC/loss_unb_alpha{}.npy'.format(puiss),loss_unb)
np.save('./results/LOGISTIC/count_unb_alpha{}.npy'.format(puiss),count_unb)

