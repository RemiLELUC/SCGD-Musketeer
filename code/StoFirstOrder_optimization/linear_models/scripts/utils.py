import numpy as np

def simu_block_ridge(seed,n_samples,n_features,puiss,block_size,noise):
    np.random.seed(seed)
    X = np.zeros((n_samples,n_features))
    for j in range(n_features//block_size):
        X_j = np.random.normal(scale=(j+1)**(-puiss),size=(n_samples,block_size))
        X[:,j*block_size:(j+1)*block_size] = X_j
    # shuffle columns of X
    idx = np.random.permutation(n_features)
    X[:, :] = X[:, idx]
    ground_truth = np.random.uniform(low=0,high=1,size=n_features)
    y = X@ground_truth
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape)
    return X, y, ground_truth

# without intercept
def simu_block_log(seed,n_samples,n_features,puiss,block_size,noise):
    np.random.seed(seed)
    X = np.zeros((n_samples,n_features))
    for j in range(n_features//block_size):
        X_j = np.random.normal(scale=(j+1)**(-puiss),size=(n_samples,block_size))
        X[:,j*block_size:(j+1)*block_size] = X_j
    # shuffle columns of X
    indices = np.arange(n_features)
    np.random.shuffle(indices)
    X[:,:] = X[:, indices]
    ground_truth = np.random.uniform(low=0,high=1,size=n_features)
    y = np.ones(n_samples)
    h = 1/(1+np.exp(-X@ground_truth))
    if noise > 0.0:
        h += np.random.normal(scale=noise, size=y.shape)
    y[h<=0.5]=-1
    return X, y

def simu_gaussian(seed,n_samples,n_features,n_informative,noise):
    np.random.seed(seed)
    X = np.random.randn(n_samples,n_features)
    ground_truth = np.zeros(n_features)
    ground_truth[:n_informative] = np.random.uniform(low=0,high=1,size=n_informative)
    y = X@ground_truth
    indices = np.arange(n_features)
    np.random.shuffle(indices)
    X[:, :] = X[:, indices]
    ground_truth = ground_truth[indices]
    # Add noise
    if noise > 0.0:
        y += np.random.normal(scale=noise, size=y.shape)
    return X, y, ground_truth

def simu_gaussian_log(seed,n_samples,n_features,n_informative,noise):
    np.random.seed(seed)
    X = np.random.randn(n_samples,n_features-1)
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept,X), axis=1)
    ground_truth = np.zeros(n_features)
    y = np.zeros(n_samples)
    ground_truth[:n_informative] = np.random.uniform(low=0,high=1,size=n_informative)
    h = 1/(1+np.exp(-X@ground_truth))
    if noise > 0.0:
        h += np.random.normal(scale=noise, size=y.shape)
    y[h>=0.5]=1
    indices = np.arange(n_features)
    np.random.shuffle(indices)
    X[:, :] = X[:, indices]
    ground_truth = ground_truth[indices]
    # Add noise
    return X, y, ground_truth