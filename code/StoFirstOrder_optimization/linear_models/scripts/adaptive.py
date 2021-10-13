import numpy as np
from scipy.stats import norm
from scipy.special import softmax

def update(moments, newValue):
    ''' Online update of count and mean according to Welford's algorithm
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    Params:
    @moments (int,float,float): count, mean, moment M2
    @newValue          (float): new value of the sequence
    Returns:
    @moments (int,float,float): updated count, mean and moment M2
    '''
    count, mean = moments
    count += 1
    delta = newValue - mean
    mean += delta / count
    return count, mean

def optimize_sum(seed,n,d,g,g_sparse,x0,N,batch_size,T,
                 a,alpha_power,eta,method,unbiased,verbose):
        ''' Perform Sparse Stochastic (Coordinate) Gradient Descent
        where X (n x d) and Y (n x 1), g(w,batch,k) is k-th coordinate of
        gradient estimate at point w 
        Params
        @seed          (int): random seed for reproducibility
        @n             (int): number of training samples
        @d             (int): dimension of the problem
        @g            (func): unbiased estimate of gradient (E[g] = \nabla f)
        @g_sparse     (func): sparse estimate of gradient
        @x0          (array): initial point of algorithm
        @N             (int): total number of iterations to perform
        @batch_size    (int): size of batch gradient estimate
        @T             (int): size of exploration
        @a           (float): numerator of the learning rate (a/k)^alpha
        @alpha_power (float): power of the learning rate (a/k)^alpha
        @method     (string): 'sgd','uni','musketeer'
        @step_constant(bool): whether to use constant or adaptive step in EXP3 alg
        @verbose      (bool): boolean to print some information 
        Returns:
        @x_list  (array): sequence of iterates of the algorithm
        '''
        np.random.seed(seed)          # set random seed
        x = x0.copy()                 # initialize iterate
        x_list = [x.copy()]
        g_info = np.zeros((d,2)) # initialize moments of the gradient
        probas = np.ones(d)/d    # initial probabilities (uniform)
        # classical sgd with dense gradient estimate
        if method=='sgd':
            for i in range(1,N+1):
                batch = np.random.choice(a=np.arange(n),size=batch_size)  
                gradient = g(x,batch)
                # update rule
                step = a * (1/i)**alpha_power
                x-= step*gradient
                # store iterate
                x_list.append(x.copy())
        # sparse sgd with uniform sampling
        elif method=='uni':
            k_list = np.random.randint(low=0,high=d,size=d*N)
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))  
            for i in range(1,d*N+1): 
                batch = batch_list[i-1]
                k = k_list[i-1]
                g_s = g_sparse(x,batch,k)
                g_info[k,0]+=1
                # update rule
                step = a * (1/((i//d)+1))**alpha_power
                x[k]-= step*g_s
                # store iterate
                if i%d==0:
                    x_list.append(x.copy())                
        elif method=='musketeer':
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))  
            tab_d = np.arange(d)
            for i in range(1,d*N+1):
                batch = batch_list[i-1] 
                # draw multinomial to select one coordinate of g
                k = np.random.choice(a=tab_d,p=probas) 
                # sparse gradient estimate (just one activate coordinate of g) 
                g_s = g_sparse(x,batch,k)
                # since g_k was drawn, we update mean for g_k
                g_info[k,:] = update(g_info[k,:],g_s/probas[k])  
                # update rule
                step = a * (1/((i//d)+1))**alpha_power
                if unbiased:
                    # reweight by probas to be unbiased
                    x[k]-= (step/d)*(g_s/probas[k])
                else:
                    x[k]-= step*g_s
                # Exploitation part; update probabilities
                if i%T==0:
                    # cumulative gain
                    G = ((i//T)*g_info[:,0]/T)*g_info[:,1]
                    G = np.abs(G)/np.linalg.norm(G,ord=np.inf)
                    if verbose:
                        print('k_sel      :',k)
                        print('(g_s/p_k)  :',(g_s/probas[k]))
                        print('G=',G)
                    # update probas
                    lbda = 1/np.log((i//T)+1)
                    probas = (1-lbda)*softmax(eta*G) + lbda*np.ones(d)/d
                    #probas = softmax(eta*G)
                    if verbose:
                        print('probas_up:',probas)
                # store current iterate
                if i%d==0:
                    x_list.append(x.copy())
        return g_info[:,0],np.array(x_list)
    