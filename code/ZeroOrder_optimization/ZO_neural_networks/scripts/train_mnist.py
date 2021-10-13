import random
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import numpy as np 
import math

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
    return None
        
class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)
        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target
    
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net, self).__init__()
        # Size of input and hidden layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # 2 linear layers
        self.linear1 = nn.Linear(self.input_size,self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size,self.output_size)
        # Dimensions for the number of parameters
        self.dim1 = self.input_size*self.hidden_size + self.hidden_size
        self.dim2 = self.hidden_size*self.hidden_size + self.hidden_size 
        self.dim3 = self.hidden_size*self.output_size + self.output_size 
        self.d = self.dim1 + self.dim2 + self.dim3

    def forward_pass(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.log_softmax(x, dim=0)
        return x
    
def train_uni(seed,batch_size,epochs,
              input_size,hidden_size,output_size,lr,h):
    with torch.no_grad():
        # set seed for reproducibility
        set_seed(seed)
        # Build Neural Network
        model = Net(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        # Put on GPU
        model.to(device)
        d = sum(p.numel() for p in model.parameters())
        # Loss function
        criterion = torch.nn.NLLLoss()
        # Initialize loss evolution
        loss_uni= []
        accuracy_list = []
        # Main training loop
        for e in range(epochs):
            running_loss = 0
            t_start = time()
            # Training Loop
            for images, labels in train_dataloader:
                # Get images,labels of current batch
                images = images.view(images.shape[0], -1)
                # Forward pass of the model at current theta
                output = model.forward_pass(images)
                # Loss function
                loss = criterion(output, labels)
                # get full vector theta
                param_vec = parameters_to_vector(model.parameters())
                param_temp = torch.clone(param_vec)
                idx_change = torch.randint(high=d,size=(1,1))[0]
                param_temp[idx_change] = param_temp[idx_change].add(h)
                # Put back to model parameters
                vector_to_parameters(param_temp,model.parameters())
                # Forward pass of the model at current theta
                output_new = model.forward_pass(images)
                # Loss function
                loss_new = criterion(output_new, labels)
                grad_curr = (loss_new-loss)/h
                # update rule
                param_vec[idx_change] = param_vec[idx_change].add(grad_curr,alpha=-lr)
                # put back into model
                vector_to_parameters(param_vec,model.parameters())
                # Store evolution of loss
                running_loss += loss.item()
                loss_uni.append(loss.item())            
            correct = 0
            total = 0
            # Testing Loop
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_list.append(correct / total)
            if e%10==0:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
                print('Accuracy :',correct / total)
        params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_batchsize'+str(batch_size)
        np.save('res_uni/loss_mnist_uni_'+params_save+'.npy',loss_uni)
        np.save('res_uni/accuracy_mnist_uni_'+params_save+'.npy',accuracy_list)
        #return loss_uni
        return None
    
def train_nes(seed,batch_size,epochs,
              input_size,hidden_size,output_size,lr,h):
    with torch.no_grad():
        # set seed for reproducibility
        set_seed(seed)
        # Build Neural Network
        model = Net(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        # Put on GPU
        model.to(device)
        d = sum(p.numel() for p in model.parameters())
        # Loss function
        criterion = torch.nn.NLLLoss()
        # Initialize loss evolution
        loss_nes = []
        accuracy_list = []
        # Main training loop
        for e in range(epochs):
            running_loss = 0
            t_start = time()
            # Training Loop
            for images, labels in train_dataloader:
                # Get images,labels of current batch
                images = images.view(images.shape[0], -1)
                # Forward pass of the model at current theta
                output = model.forward_pass(images)
                # Loss function
                loss = criterion(output, labels)
                # get full vector theta
                param_vec = parameters_to_vector(model.parameters())
                param_temp = torch.clone(param_vec)
                u = torch.randn(d).to(device)
                u/= torch.norm(u)
                param_temp = param_temp.add(h*u)
                # Put back to model parameters
                vector_to_parameters(param_temp,model.parameters())
                # Forward pass of the model at current theta
                output_new = model.forward_pass(images)
                # Loss function
                loss_new = criterion(output_new, labels)
                grad_curr = (loss_new-loss)/h
                # update rule
                param_vec = param_vec.add(grad_curr*u,alpha=-lr)
                # put back into model
                vector_to_parameters(param_vec,model.parameters())
                # Store evolution of loss
                running_loss += loss.item()
                loss_nes.append(loss.item())            
            correct = 0
            total = 0
            # Testing Loop
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_list.append(correct / total)
            if e%10==0:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
                print('Accuracy :',correct / total)
        params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_batchsize'+str(batch_size)
        np.save('res_nes/loss_mnist_nes_'+params_save+'.npy',loss_nes)
        np.save('res_nes/accuracy_mnist_nes_'+params_save+'.npy',accuracy_list)
        #return loss_nes
        return None

def train_avg(seed,batch_size,epochs,
              input_size,hidden_size,output_size,
              T,lr,h):
    with torch.no_grad():
        # set seed for reproducibility
        set_seed(seed)
        # Build Neural Network
        model = Net(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        # Put on GPU
        model.to(device)
        d = sum(p.numel() for p in model.parameters())
        g_info = torch.zeros(d,2).to(device)
        ones = torch.ones(d).to(device)
        probas = torch.ones(d).to(device)/d
        G = torch.zeros(d).to(device)
        cpt = torch.zeros(1).to(device)
        criterion = torch.nn.NLLLoss()
        # Initialize loss evolution
        loss_mus= []
        accuracy_list = []
        for e in range(epochs):
            running_loss = 0
            t_start = time()
            # Training Loop
            for images, labels in train_dataloader:
                cpt +=1
                # Get images,labels of current batch
                images = images.view(images.shape[0], -1)
                # Forward pass of the model at current theta
                output = model.forward_pass(images)
                # Loss function
                loss = criterion(output, labels)
                # get full vector theta
                param_vec = parameters_to_vector(model.parameters())
                param_temp = torch.clone(param_vec)
                # draw random coordinate
                k = torch.multinomial(input=probas,num_samples=1)
                param_temp[k] = param_temp[k].add(h)
                # Put back to model parameters
                vector_to_parameters(param_temp,model.parameters())
                # Forward pass of the model at new theta
                output_new = model.forward_pass(images)
                # Loss function at new theta
                loss_new = criterion(output_new, labels)
                # Compute ZO gradient estimate
                grad_curr = (loss_new-loss)/h
                # Update counts and cumulative Gains
                g_info[k,0] +=1
                g_info[k,1] += grad_curr.to("cpu")/probas[k]
                # update rule of parameter theta
                param_vec[k] = param_vec[k].add(grad_curr,alpha=-lr)
                # put back into model
                vector_to_parameters(param_vec,model.parameters())
                # Store evolution of loss
                running_loss += loss.item()
                loss_mus.append(loss.item())
                # Update probabilities (Exploitation Part)
                if cpt%T==0:
                    G_T = g_info[:,1]/T
                    diff = G_T - G
                    G += diff/(cpt//T)
                    if torch.sum(G)==0:
                        probas = ones/d
                    else:
                        lbda = (1/torch.log((cpt//T)+3))
                        probas = (1-lbda)*(torch.abs(G)/torch.sum(torch.abs(G))) + lbda*ones/d
                        #probas = (1-lbda)*torch.softmax(G,dim=0).to(device) + lbda*torch.ones(d).to(device)/d               
            correct = 0
            total = 0
            # Testing Loop
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_list.append(correct / total)
            if e%10==0:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
                print('Accuracy :',correct / total)
        params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_batchsize'+str(batch_size)
        np.save('res_avg/loss_mnist_avg_'+params_save+'.npy',loss_mus)
        np.save('res_avg/accuracy_mnist_avg_'+params_save+'.npy',accuracy_list)
        return None
    
def train_sqr(seed,batch_size,epochs,
              input_size,hidden_size,output_size,
              T,lr,h):
    with torch.no_grad():
        # set seed for reproducibility
        set_seed(seed)
        # Build Neural Network
        model = Net(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        # Put on GPU
        model.to(device)
        d = sum(p.numel() for p in model.parameters())
        g_info = torch.zeros(d,2).to(device)
        ones = torch.ones(d).to(device)
        probas = torch.ones(d).to(device)/d
        G = torch.zeros(d).to(device)
        cpt = torch.zeros(1).to(device)
        # Loss function
        criterion = torch.nn.NLLLoss()
        # Initialize loss evolution
        loss_mus= []
        accuracy_list = []
        for e in range(epochs):
            running_loss = 0
            t_start = time()
            # Training Loop
            for images, labels in train_dataloader:
                cpt +=1
                # Get images,labels of current batch
                images = images.view(images.shape[0], -1)
                # Forward pass of the model at current theta
                output = model.forward_pass(images)
                # Loss function
                loss = criterion(output, labels)
                # get full vector theta
                param_vec = parameters_to_vector(model.parameters())
                param_temp = torch.clone(param_vec)
                # draw random coordinate
                k = torch.multinomial(input=probas,num_samples=1)
                param_temp[k] = param_temp[k].add(h)
                # Put back to model parameters
                vector_to_parameters(param_temp,model.parameters())
                # Forward pass of the model at new theta
                output_new = model.forward_pass(images)
                # Loss function at new theta
                loss_new = criterion(output_new, labels)
                # Compute ZO gradient estimate
                grad_curr = (loss_new-loss)/h
                # Update counts and cumulative Gains
                g_info[k,0] +=1
                g_info[k,1] += (grad_curr.to("cpu"))**2/probas[k]
                # update rule of parameter theta
                param_vec[k] = param_vec[k].add(grad_curr,alpha=-lr)
                # put back into model
                vector_to_parameters(param_vec,model.parameters())
                # Store evolution of loss
                running_loss += loss.item()
                loss_mus.append(loss.item())
                # Update probabilities (Exploitation Part)
                if cpt%T==0:
                    G_T = g_info[:,1]/T
                    diff = G_T - G
                    G += diff/(cpt//T)
                    if torch.sum(G)==0:
                        probas = ones/d
                    else:
                        lbda = (1/torch.log((cpt//T)+3))
                        probas = (1-lbda)*(torch.abs(G)/torch.sum(torch.abs(G))) + lbda*ones/d
                        #probas = (1-lbda)*torch.softmax(G,dim=0).to(device) + lbda*torch.ones(d).to(device)/d               
            correct = 0
            total = 0
            # Testing Loop
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_list.append(correct / total)
            if e%10==0:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
                print('Accuracy :',correct / total)
        params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_batchsize'+str(batch_size)
        np.save('res_sqr/loss_mnist_sqr_'+params_save+'.npy',loss_mus)
        np.save('res_sqr/accuracy_mnist_sqr_'+params_save+'.npy',accuracy_list)
        return None
    
def train_abs(seed,batch_size,epochs,
              input_size,hidden_size,output_size,
              T,lr,h):
    with torch.no_grad():
        # set seed for reproducibility
        set_seed(seed)
        # Build Neural Network
        model = Net(input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size)
        # Put on GPU
        model.to(device)
        d = sum(p.numel() for p in model.parameters())
        g_info = torch.zeros(d,2).to(device)
        ones = torch.ones(d).to(device)
        probas = torch.ones(d).to(device)/d
        G = torch.zeros(d).to(device)
        cpt = torch.zeros(1).to(device)
        # Loss function
        criterion = torch.nn.NLLLoss()
        # Initialize loss evolution
        loss_mus= []
        accuracy_list = []
        for e in range(epochs):
            running_loss = 0
            t_start = time()
            # Training Loop
            for images, labels in train_dataloader:
                cpt +=1
                # Get images,labels of current batch
                images = images.view(images.shape[0], -1)
                # Forward pass of the model at current theta
                output = model.forward_pass(images)
                # Loss function
                loss = criterion(output, labels)
                # get full vector theta
                param_vec = parameters_to_vector(model.parameters())
                param_temp = torch.clone(param_vec)
                # draw random coordinate
                k = torch.multinomial(input=probas,num_samples=1)
                param_temp[k] = param_temp[k].add(h)
                # Put back to model parameters
                vector_to_parameters(param_temp,model.parameters())
                # Forward pass of the model at new theta
                output_new = model.forward_pass(images)
                # Loss function at new theta
                loss_new = criterion(output_new, labels)
                # Compute ZO gradient estimate
                grad_curr = (loss_new-loss)/h
                # Update counts and cumulative Gains
                g_info[k,0] +=1
                g_info[k,1] += torch.abs(grad_curr).to("cpu")/probas[k]
                # update rule of parameter theta
                param_vec[k] = param_vec[k].add(grad_curr,alpha=-lr)
                # put back into model
                vector_to_parameters(param_vec,model.parameters())
                # Store evolution of loss
                running_loss += loss.item()
                loss_mus.append(loss.item())
                # Update probabilities (Exploitation Part)
                if cpt%T==0:
                    G_T = g_info[:,1]/T
                    diff = G_T - G
                    G += diff/(cpt//T)
                    if torch.sum(G)==0:
                        probas = ones/d
                    else:
                        lbda = (1/torch.log((cpt//T)+3))
                        probas = (1-lbda)*(torch.abs(G)/torch.sum(torch.abs(G))) + lbda*ones/d
                        #probas = (1-lbda)*torch.softmax(G,dim=0).to(device) + lbda*torch.ones(d).to(device)/d               
            correct = 0
            total = 0
            # Testing Loop
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy_list.append(correct / total)
            if e%10==0:
                print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
                print('Accuracy :',correct / total)
        params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)+'_batchsize'+str(batch_size)
        np.save('res_abs/loss_mnist_abs_'+params_save+'.npy',loss_mus)
        np.save('res_abs/accuracy_mnist_abs_'+params_save+'.npy',accuracy_list)
        return None
    
    
########## Parameter Configuration ##########
batch_size = 32
input_size = 28*28
hidden_size = 32
output_size = 64
lr = 10
h = 0.01
epochs = 100
# Exploration size
T = torch.Tensor([234]).to(device)
#############################################


############ LOAD DATASET ###################
# Load data
train_dataset = FastMNIST('data/MNIST', train=True, download=True)
test_dataset = FastMNIST('data/MNIST', train=False, download=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
#############################################

# Uniform Coordinate Sampling
for seed in range(1,11):
    print('seed=',seed)
    train_uni(seed=seed,batch_size=batch_size,epochs=epochs,
          input_size=input_size,hidden_size=hidden_size,output_size=output_size,
          lr=lr,h=h)
# Gaussian smoothing estimate (Nesterov/Spokoiny)
for seed in range(1,11):
    print('seed=',seed)
    train_nes(seed=seed,batch_size=batch_size,epochs=epochs,
          input_size=input_size,hidden_size=hidden_size,output_size=output_size,
          lr=lr,h=h)
!zip -r /content/res_nes.zip /content/res_nes

# Musketeer average gains
for seed in range(1,11):
    print('seed=',seed)
    train_avg(seed=seed,batch_size=batch_size,epochs=epochs,
          input_size=input_size,hidden_size=hidden_size,output_size=output_size,
          T=T,lr=lr,h=h)
    
# Musketeer square gains
for seed in range(1,11):
    print('seed=',seed)
    train_sqr(seed=seed,batch_size=batch_size,epochs=epochs,
          input_size=input_size,hidden_size=hidden_size,output_size=output_size,
          T=T,lr=lr,h=h)
    
# Musketeer absolute value gains
for seed in range(1,11):
    print('seed=',seed)
    train_abs(seed=seed,batch_size=batch_size,epochs=epochs,
          input_size=input_size,hidden_size=hidden_size,output_size=output_size,
          T=T,lr=lr,h=h)
