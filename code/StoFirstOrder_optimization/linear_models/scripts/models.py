import numpy as np
from adaptive import optimize_sum

class linearReg:
    def __init__(self, X, y, 𝜆):
        self.X = X  # data matrix
        self.y = y  # regression labels
        self.𝜆 = 𝜆
        self.n = X.shape[0] # n_samples
        self.d = X.shape[1] # n_features

    def loss(self,w):
        ''' Loss function OLS: L(w) = (1/2n) || Y - Xw ||^2  + (𝜆/2) ||w||^2'''
        data_term = np.sum((self.y-self.X.dot(w))**2)/(2*self.n)
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg

    def batch_grad(self,w,batch):
        ''' Batch SG: g(w) = (1/|B|) X_b^T (X_bw-Y_b) '''
        X_b = self.X[batch]
        err = X_b.dot(w)-self.y[batch]
        grad = (X_b.T).dot(err)
        return grad/len(batch) + self.𝜆*w

    def sparse_grad(self,w,batch,k):
        ''' Coordinate Batch SG: g(w)_k = (1/|B|) {X_b^T (X_bw-Y_b)}_k '''
        X_b = self.X[batch]
        err = X_b.dot(w)-self.y[batch]
        grad = X_b[:,k].dot(err)
        return grad/len(batch) + self.𝜆*w[k]
    
    def sparse_hess(self,w,batch,k):
        ''' Coordinate Batch Hessian: g(w)_k = (1/|B|) {X_b^T (X_bw-Y_b)}_k '''
        X_b = self.X[batch]
        hess = (X_b[:,k].T).dot(X_b[:,k])
        return hess/len(batch) + self.𝜆*np.eye(self.d)

    def fit(self,seed,x0,N,batch_size,T,a,alpha_power,eta,
            method,unbiased,verbose):
        ''' Fit model with Stochastic Gradient methods '''
        # Linear Regression Model
        return optimize_sum(seed=seed,n=self.n,d=self.d,
                            g=self.batch_grad,g_sparse=self.sparse_grad,
                            x0=x0,N=N,batch_size=batch_size,T=T,
                            a=a,alpha_power=alpha_power,eta=eta,
                            method=method,unbiased=unbiased,
                            verbose=verbose)
    
class logisticReg:
    def __init__(self, X, y, 𝜆, fit_intercept=True):
        self.X = X               # data matrix
        self.y = y               # regression labels
        self.𝜆 = 𝜆
        self.n = X.shape[0]
        self.fit_intercept = fit_intercept
        if self.fit_intercept:
            self.add_intercept()
        self.d = (self.X).shape[1]
                    
    def add_intercept(self):
        ''' Add column of 1 for intercept '''
        intercept = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate((intercept, self.X), axis=1)
        
    def sigmoid(self, z):
        ''' Sigmoid function '''
        return 1 / (1 + np.exp(-z))
    
    def loss(self,w):
        ''' Regularized objective function at point w 
        for i=1,...,n, proba \pi_i = sigmoid(X_i^T w)
        loss(w) = -(1/n) \sum_{i=1}^n [y_i log(\pi_i) + (1-y_i) log(1-\pi_i)] + (𝜆/2)|w|^2
        Params:
        @w (array px1): point to evaluate
        Returns:
        @loss  (float): penalized loss function at w
        '''
        z = np.dot(self.X,w)
        data_term = np.log(1+np.exp(np.multiply(-self.y,z))).mean()
        # Regulatization term (Ridge penalization)
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg

    def batch_grad(self,w,batch):
        ''' Batch Gradient of regularized objective function
        Params:
        @batch (array Bx1): indices of the batch
        @w     (array px1): point to evaluate
        Returns:
        @gradient_batch (array px1): batch gradient at w
        '''           
        B = len(batch)
        X_b = self.X[batch]
        z_b = np.dot(X_b, w)
        h_b = self.sigmoid(z_b)
        err = h_b - self.y[batch]
        g = (1/B)*(X_b.T).dot(err)
        if self.fit_intercept:
            g0 = g[0]
            g1 = g[1:] + self.𝜆*w[1:]
            gradient_batch = np.vstack((g0.reshape(-1,1),g1.reshape(-1,1))).ravel()
        else:
            gradient_batch = g + self.𝜆*w
        return gradient_batch
    
    def sparse_grad(self,w,batch,k):
        ''' Batch Gradient of regularized objective function
        Params:
        @batch (array Bx1): indices of the batch
        @w     (array px1): point to evaluate
        Returns:
        @gradient_batch (array px1): batch gradient at w
        '''
        B = len(batch)
        X_b = self.X[batch]
        z_b = np.dot(self.X[batch], w)
        h_b = self.sigmoid(z_b)
        err = h_b - self.y[batch]
        g = (1/B)*X_b[:,k].dot(err)
        if self.fit_intercept:
            if k==0:
                res = g
            else:
                res = g + self.𝜆*w[k]
        else:
            res = g + self.𝜆*w[k]
        return res
    
    def batch_hessian(self,batch,w):
        ''' Batch Hessian of regularized objective function
        Params:
        @batch   (array Bx1): indices of the batch
        @w       (array px1): point to evaluate
        Returns:
        @H_batch (array pxp): batch hessian at w
        '''
        Ip = np.eye(self.X.shape[1])
        B = len(batch)
        if B==0:
            H_batch = Ip
        else:
            X_b = self.X[batch]
            z_b = np.dot(X_b, w)
            h_b = self.sigmoid(z_b)
            H_batch = (1/B)*np.dot(np.dot(X_b.T,np.diag(h_b*(1-h_b))),X_b) + self.𝜆*Ip
            H_batch[0,0] -= self.𝜆
        return H_batch
    

    def fit(self,seed,x0,N,batch_size,T,a,alpha_power,eta,
            method,unbiased,verbose):
        ''' Fit model with Stochastic Gradient methods '''
        # Logistic Regression Model
        return optimize_sum(seed=seed,n=self.n,d=self.d,
                            g=self.batch_grad,g_sparse=self.sparse_grad,
                            x0=x0,N=N,batch_size=batch_size,T=T,
                            a=a,alpha_power=alpha_power,eta=eta,
                            method=method,unbiased=unbiased,
                            verbose=verbose)
    
    def predict_prob(self,X_test,w):
        ''' Predict probabilities given X_test: y_pred = sigmoid(X^T w_final)
        Params:
        @X_test (array n_test x p): data to predict
        Returns
        @y_pred (array n_test x 1): probabilities array
        '''
        if self.fit_intercept:
            intercept = np.ones((X_test.shape[0], 1))
            X_pred = np.concatenate((intercept, X_test), axis=1)
        else:
            X_pred = X_test
        return self.sigmoid(np.dot(X_pred, w))
    
    def predict(self,X_test,w,threshold=0.5):
        ''' Predict binary labels given X_test
        Params:
        @X_test (array n_test x p): data to predict
        @threshold (float in O,1): threshold for classification, default 0.5
        Returns
        @y_pred (array n_test x 1): binary array
        '''
        return (self.predict_prob(X_test=X_test,w=w) >= threshold).astype(int) 