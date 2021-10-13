import torch 
import random
import numpy as np
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils import parameters_to_vector, vector_to_parameters

if torch.cuda.is_available():  
    dev = "cuda" 
else:  
    dev = "cpu"  
print(dev)
device = torch.device(dev)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
# tool function for musketeer optimizer
def update(moments, newValues):
    counts, means = moments[:,0],moments[:,1]
    counts.add_(1)
    deltas = newValues.sub(means)
    means.add_(deltas.div(counts))
    res = torch.dstack([counts,means])[0]
    return res

class Musketeer(Optimizer):
    r"""Implements MUSKETEER algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
                           parameter groups
        T              (int): Exploration size
        ratio_changes(float): percentage of changes in coordinates at each update
        lr           (float): learning rate
        eta          (float): weight parameter in softmax operator
    """
    def __init__(self, params, T, 
                 ratio_changes=0.1, lr=required, eta=1):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        d = sum(p.numel() for p in params)
        count = 0
        g_info = torch.zeros(d,2).to(device)
        probas = torch.ones(d).to(device)/d
        nb_changes = int(ratio_changes*d)
        defaults = dict(lr=lr,eta=eta,T=T,d=d,count=count,
                        probas=probas,g_info=g_info,nb_changes=nb_changes)
        super(Musketeer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # Update count
        self.param_groups[0]['count'] += self.param_groups[0]['nb_changes']
        #print(count)
        # Loss
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ### Exploration Part ###
        param_vec = parameters_to_vector(self.param_groups[0]['params'])
        # draw random coordinates to update
        idx_changes = torch.multinomial(input=self.param_groups[0]['probas'],
                                        num_samples=self.param_groups[0]['nb_changes']).to(device)
        # compute gradients
        grads = []
        for param in self.param_groups[0]['params']:
            grads.append(param.grad.view(-1))
        # Gradients for updates
        grads = torch.cat(grads)[idx_changes]
        # Update parameters
        param_vec[idx_changes] = param_vec[idx_changes].add(grads,alpha=-self.param_groups[0]['lr'])
        # Update means and variance for these gradients
        self.param_groups[0]['g_info'][idx_changes] = update(self.param_groups[0]['g_info'][idx_changes],grads/self.param_groups[0]['probas'][idx_changes])
        # Put back to model parameters
        vector_to_parameters(param_vec,self.param_groups[0]['params'])
        # Exploitation Part
        if self.param_groups[0]['count']>self.param_groups[0]['T']:
            G = (torch.abs(self.param_groups[0]['g_info'][:,1])/torch.linalg.norm(self.param_groups[0]['g_info'][:,1],ord=float("Inf"))).to(device)
            self.param_groups[0]['probas']  = torch.exp(self.param_groups[0]['eta']*G)/(torch.exp(self.param_groups[0]['eta']*G).sum())
            self.param_groups[0]['count']=0
        return loss