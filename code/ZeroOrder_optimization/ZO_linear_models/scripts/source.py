import numpy as np
from scipy.stats import norm
from scipy.special import softmax

def vector_basis(size,index):
    res = np.zeros(size)
    res[index]=1.0
    return res

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

def optimize(seed,n,d,f,batch_size,
             gamma,mu_power,x0,N,T,
             a,t0,alpha_power,method,
             importance,gains,
             fixed,eta,verbose):
        ''' Perform Sparse Stochastic (Coordinate) Gradient Descent
        on noisy function f that can be approximate with a batch-size 
        Params
        @seed          (int): random seed for reproducibility
        @n             (int): number of training samples
        @d             (int): dimension of the problem
        @f            (func): objective function
        @batch_size    (int): size of batch gradient estimate
        @gamma       (float): numerator of smoothing parameter
        @mu_power    (float): power denominator smoothing parameter
        @x0          (array): initial point of algorithm
        @N             (int): total number of iterations to perform
        @T             (int): size of exploration
        @a           (float): numerator of the learning rate (a/k)^alpha
        @alpha_power (float): power of the learning rate (a/k)^alpha
        @method     (string): 'full','uni','musketeer'
        @fixed        (bool): whether to use constant or adaptive step in EXP3 alg
        @eta         (float): mixture parameter when fixed=True, in [0,1]
        @verbose      (bool): boolean to print some information 
        Returns:
        @x_list  (array): sequence of iterates of the algorithm
        '''
        np.random.seed(seed)          # set random seed
        x = x0.copy()                 # initialize iterate
        x_list = [x.copy()]
        g_info = np.zeros((d,2)) # initialize moments of the gradient
        ones = np.ones(d)
        probas = np.ones(d)/d    # initial probabilities (uniform)
        # classical sgd with dense gradient estimate
        if method=='full':
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))
            for i in range(1,N+1):
                batch_curr = batch_list[(i-1)*d:i*d,:] 
                #if i%10==0:
                #    a/=2
                mu = gamma * (1/i)**mu_power
                gradient = np.zeros(d)
                for k in range(d): # compute full gradient estimate
                    e_k = np.zeros(d)
                    e_k[k] = 1.0
                    gradient[k] = (f(batch_curr[k],x+mu*e_k)-f(batch_curr[k],x))/mu
                # update rule
                step = a * (1/(i+t0))**alpha_power
                x-= step*gradient
                # store iterate
                x_list.append(x.copy())
        # sparse sgd with uniform sampling
        elif method=='uni':
            # batch_list for the samples
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))  
            # coordinates list for the coordinates to draw
            k_list = np.random.randint(low=0,high=d,size=d*N)
            for i in range(1,d*N+1): 
                batch = batch_list[i-1] 
                mu = gamma * (1/((i//d)+1))**mu_power
                #if i%(10*d)==0:
                #    a/=2
                k = k_list[i-1]
                e_k = np.zeros(d)
                e_k[k] = 1.0
                # gradient estimate at coordinate k
                g_s = (f(batch,x+mu*e_k)-f(batch,x))/mu
                g_info[k,0]+=1
                # update rule
                step = a * (1/((i//d)+t0))**alpha_power
                x[k]-= step*g_s
                # store iterate
                if i%d==0:
                    x_list.append(x.copy()) 
        elif method=='nes':
            # batch_list for the samples
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))  
            for i in range(1,d*N+1): 
                batch = batch_list[i-1] 
                mu = gamma * (1/((i//d)+1))**mu_power
                #if i%(10*d)==0:
                #    a/=2
                u = np.random.randn(d)
                u /= np.linalg.norm(u)
                # gradient estimate at coordinate k
                g_s = (f(batch,x+mu*u)-f(batch,x))/mu
                # update rule
                step = a * (1/((i//d)+t0))**alpha_power
                x-= step*g_s*u
                # store iterate
                if i%d==0:
                    x_list.append(x.copy()) 
        elif method=='mus': 
            # batch_list for the samples
            batch_list = np.random.choice(a=np.arange(n),size=(d*N,batch_size))
            # Total gains for probabilities updates
            G = np.zeros(d)
            tab_d = np.arange(d)
            # Gains map function for identity,absolute or square
            if gains=='avg':
                f_gain = lambda x: x
            if gains=='abs':
                f_gain = lambda x: np.abs(x)
            if gains=='square':
                f_gain = lambda x: x**2
            # Main loop
            for i in range(1,d*N+1):
                batch = batch_list[i-1] 
                # smoothing parameter for gradient approximation
                mu = gamma * (1/((i//d)+1))**mu_power
                #if i%(10*d)==0:
                #    a/=2
                # draw coordinates for exploration phase
                k = np.random.choice(a=tab_d,p=probas) 
                # sparse gradient estimate (just one activate coordinate of g) 
                e_k = np.zeros(d)
                e_k[k] = 1.0
                # gradient estimate at coordinate k
                g_s = (f(batch,x+mu*e_k)-f(batch,x))/mu
                # since g_k was drawn, we update mean for g_k
                g_info[k,0]+=1
                # update cumulative gains
                g_info[k,1]+=f_gain(g_s)/probas[k]  
                # update rule
                step = a * (1/((i//d)+t0))**alpha_power
                if importance: 
                    # reweight by probas for importance sampling
                    x[k]-= (step/d)*(g_s/probas[k])
                else:
                    x[k]-= step*g_s
                # Exploitation part; update probabilities
                if i%T==0:
                    # average gain during exploration
                    G_T = g_info[:,1]/T
                    # cumulative gain
                    diff = G_T-G
                    G+= diff/(i//T)
                    #G = ((i//T)*g_info[:,0]/T)*g_info[:,1]
                    #G = np.abs(G)/np.linalg.norm(G,ord=np.inf)
                    if verbose:
                        print('k_sel      :',k)
                        print('(g_s/p_k)  :',(g_s/probas[k]))
                        print('G=',G)
                    # update probas
                    if np.sum(G)==0:
                            probas = ones/d
                    else:
                        #G = np.abs(G)/np.linalg.norm(G,ord=np.inf)
                        if fixed:
                            probas = (1-eta)*(np.abs(G)/np.sum(np.abs(G))) + eta*ones/d
                        else:
                            lbda = 1/np.log((i//T)+3)
                            probas = (1-lbda)*(np.abs(G)/np.sum(np.abs(G))) + lbda*ones/d
                    #else:
                    #    if np.sum(G)==0:
                    #        probas = ones/d
                    #    else:
                    #        if fixed:
                    #            probas = (1-eta)*(G/np.sum(G)) + eta*ones/d
                                #probas = (1-eta)*softmax(G) + eta*ones/d
                    #        else:
                    #            lbda = 1/np.log((i//T)+3)
                    #            probas = (1-lbda)*(G/np.sum(G)) + lbda*ones/d
                                #probas = (1-lbda)*softmax(G) + lbda*ones/d
                                #if verbose:
                                #    print('lbda=',lbda)
                        #probas = (1-lbda)*softmax(eta*G) + lbda*np.ones(d)/d
                        #probas = softmax(eta*G)
                        if verbose:
                            print('probas_up:',probas)
                # store current iterate
                if i%d==0:
                    x_list.append(x.copy())     
        return g_info[:,0],np.array(x_list)
    