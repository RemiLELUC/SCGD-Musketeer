import numpy as np
from utils import simu_block_ridge
from simus import run_ols

# Parameters
n_samples = 10000   # number of samples
n_features = 250    # dimension of the problem
# Simulate data for regression
seed=0
puiss=10
noise=0.01
block_size=1

X,y,coeff=simu_block_ridge(seed=seed,n_samples=n_samples,n_features=n_features,
                     puiss=puiss,block_size=block_size,noise=noise)

𝜆 = 1/n_samples          #regularization parameter
G = ((X.T)@X)/n_samples  # Gram matrix
A = G + 𝜆*np.eye(n_features)
B = ((X.T)@y)/n_samples
ridge = np.linalg.solve(a=A ,b=B)

data_opt = np.sum((y-X.dot(ridge))**2)/(2*n_samples)
reg_opt = (𝜆/2) * sum(ridge**2)
loss_opt = data_opt + reg_opt
print('data_opt:',data_opt)
print('reg_opt :',reg_opt)
print(loss_opt)

N = int(1e2)    # number of passes over coordinates
a = 1           # numerator of learning rate
alpha_power = 1 # power in the learning rate
batch_size =  8   # batch size for gradient estimates
verbose = False # to display information
N_exp = 20     # number of experiments
eta = 1
T = int(np.sqrt(n_features)) # size of exploration
print('T  :',T)
print('𝜆  :',𝜆)
print('eta:',eta)

# adaptive
_,_,loss_sgd = run_ols(X=X,y=y,𝜆=𝜆,method='sgd',N_exp=N_exp,N=N,
                       batch_size=batch_size,T=None,a=a,alpha_power=alpha_power,
                       eta=None,unbiased=None,verbose=False)
np.save('./results/RIDGE/mean_sgd_alpha{}.npy'.format(puiss),np.mean(loss_sgd,axis=0))

_,count_uni,loss_uni = run_ols(X=X,y=y,𝜆=𝜆,method='uni',N_exp=N_exp,N=N,
                                batch_size=batch_size,T=None,a=a,alpha_power=alpha_power,
                                eta=None,unbiased=None,verbose=False)
np.save('./results/RIDGE/mean_uni_alpha{}.npy'.format(puiss),np.mean(loss_uni,axis=0))
np.save('./results/RIDGE/count_uni_alpha{}.npy'.format(puiss),count_uni)

_,count_bia,loss_bia = run_ols(X=X,y=y,𝜆=𝜆,method='musketeer',N_exp=N_exp,N=N,
                                batch_size=batch_size,T=T,a=a,alpha_power=alpha_power,
                                eta=eta,unbiased=False,verbose=False)
np.save('./results/RIDGE/mean_bia_alpha{}.npy'.format(puiss),np.mean(loss_bia,axis=0))
np.save('./results/RIDGE/count_bia_alpha{}.npy'.format(puiss),count_bia)

_,count_unb,loss_unb= run_ols(X=X,y=y,𝜆=𝜆,method='musketeer',N_exp=N_exp,N=N,
                               batch_size=batch_size,T=T,a=a,alpha_power=alpha_power,
                               eta=eta,unbiased=True,verbose=False)
np.save('./results/RIDGE/mean_unb_alpha{}.npy'.format(puiss),np.mean(loss_unb,axis=0))
np.save('./results/RIDGE/count_unb_alpha{}.npy'.format(puiss),count_unb)

