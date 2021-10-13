import numpy as np
from source import optimize

class my_svm:
    def __init__(self, X, y, 𝜆):
        self.X = X  # data matrix
        self.y = y  # regression labels
        self.𝜆 = 𝜆
        self.n = X.shape[0] # n_samples
        self.d = X.shape[1] # n_features
        
    def loss(self,batch,w):
        ''' Loss function SVM: L(w) = (1/2n) || Y - Xw ||^2  + (𝜆/2) ||w||^2'''
        X_b = self.X[batch]
        data_term = 1-self.y[batch]*X_b.dot(w)
        data_term[data_term<0]=0
        reg = (self.𝜆/2) * sum(w**2)
        return np.sum(data_term)/len(batch) + reg 

    def fit(self,seed,x0,N,batch_size,T,gamma,mu_power,
            a,t0,alpha_power,fixed,eta,method,importance,gains,
            verbose):
        ''' Fit model with ZO-optimization methods '''
        # Linear Regression Model
        return optimize(seed=seed,n=self.n,d=self.d,
                        f=self.loss,batch_size=batch_size,
                        gamma=gamma,mu_power=mu_power,
                        x0=x0,N=N,T=T,a=a,t0=t0,alpha_power=alpha_power,
                        fixed=fixed,eta=eta,method=method,
                        importance=importance,gains=gains,
                        verbose=verbose)

class linearReg:
    def __init__(self, X, y, 𝜆):
        self.X = X  # data matrix
        self.y = y  # regression labels
        self.𝜆 = 𝜆
        self.n = X.shape[0] # n_samples
        self.d = X.shape[1] # n_features
        
    def loss(self,batch,w):
        ''' Loss function OLS: L(w) = (1/2n) || Y - Xw ||^2  + (𝜆/2) ||w||^2'''
        X_b = self.X[batch]
        data_term = np.sum((self.y[batch]-X_b.dot(w))**2)/(2*len(batch))
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg 

    def fit(self,seed,x0,N,batch_size,T,gamma,mu_power,
            a,t0,alpha_power,fixed,eta,method,importance,gains,
            verbose):
        ''' Fit model with ZO-optimization methods '''
        # Linear Regression Model
        return optimize(seed=seed,n=self.n,d=self.d,
                        f=self.loss,batch_size=batch_size,
                        gamma=gamma,mu_power=mu_power,
                        x0=x0,N=N,T=T,a=a,t0=t0,alpha_power=alpha_power,
                        fixed=fixed,eta=eta,method=method,
                        importance=importance,gains=gains,
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
    
    def loss(self,batch,w):
        ''' Regularized objective function at point w 
        for i=1,...,n, proba \pi_i = sigmoid(X_i^T w)
        Params:
        @w (array px1): point to evaluate
        Returns:
        @loss  (float): penalized loss function at w
        '''
        z = np.dot(self.X[batch],w)
        data_term = np.log(1+np.exp(np.multiply(-self.y[batch],z))).mean()
        # Regulatization term (Ridge penalization)
        reg = (self.𝜆/2) * sum(w**2)
        return data_term + reg
 
    def fit(self,seed,x0,N,batch_size,T,gamma,mu_power,
            a,t0,alpha_power,fixed,eta,method,importance,gains,
            verbose):
        ''' Fit model with ZO-optimization methods '''
        # Logistic Regression Model
        return optimize(seed=seed,n=self.n,d=self.d,
                        f=self.loss,batch_size=batch_size,
                        gamma=gamma,mu_power=mu_power,
                        x0=x0,N=N,T=T,a=a,t0=t0,alpha_power=alpha_power,
                        fixed=fixed,eta=eta,method=method,
                        importance=importance,gains=gains,
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