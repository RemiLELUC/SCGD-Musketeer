import random
import torch 
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from musketeer_optimizer import Musketeer
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
    
def train_sgd(seed,batch_size,epochs,
              input_size,hidden_size,output_size,lr):
    # set seed for reproducibility
    set_seed(seed)
    # Load data
    train_dataset = FastMNIST('data/MNIST', train=True, download=True)
    test_dataset = FastMNIST('data/MNIST', train=False, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
    # Build Neural Network
    model = Net(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
    # Put on GPU
    model.to(device)
    # Loss function
    criterion = torch.nn.NLLLoss()
    # Optimizer SGD
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    # Initialize loss evolution
    loss_sgd= []
    accuracy_list = []
    # Main training loop
    for e in range(epochs):
        running_loss = 0
        t_start = time()
        for images, labels in train_dataloader:
            # Set gradients to 0
            optimizer.zero_grad()
            # Get images,labels of current batch
            images = images.view(images.shape[0], -1)
            # Forward pass of the model
            output = model.forward_pass(images)
            # Loss function
            loss = criterion(output, labels)
            # Compute gradients
            loss.backward()
            # Perform one step of SGD optimizer
            optimizer.step()
            # Store evolution of loss
            running_loss += loss.item()
            loss_sgd.append(loss.item())
        #print("Time this epoch:",time()-t_start)
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
        correct = 0
        total = 0
        with torch.no_grad():
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy_list.append(correct / total)
    params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)
    np.save('sgd/loss_mnist_sgd_'+params_save+'.npy',loss_sgd)
    np.save('sgd/accuracy_mnist_sgd_'+params_save+'.npy',accuracy_list)
    return None
        
def train_musketeer(seed,batch_size,epochs,input_size,hidden_size,output_size,lr,ratio_changes,eta):
    # set seed for reproducibility
    set_seed(seed)
    # Load data
    train_dataset = FastMNIST('data/MNIST', train=True, download=True)
    test_dataset = FastMNIST('data/MNIST', train=False, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=10000, shuffle=False, num_workers=0)
    # Build Neural Network
    model = Net(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
    # Put on GPU
    model.to(device)
    # Loss function
    criterion = torch.nn.NLLLoss()
    # Optimizer Musketeer
    # exploration size
    T = int(math.sqrt(model.d))
    optimizer = Musketeer(params=list(model.parameters()),T=T,lr=lr,
                          ratio_changes=ratio_changes,eta=eta)
    # Initialize loss evolution
    loss_musketeer = []
    accuracy_list = []
    # Main training loop
    for e in range(int(epochs/ratio_changes)):
        running_loss = 0
        t_start = time()
        for images, labels in train_dataloader:
            # Set gradients to 0
            optimizer.zero_grad()
            # Get images,labels of current batch
            images = images.view(images.shape[0], -1)
            # Forward pass of the model
            output = model.forward_pass(images)
            # Loss function
            loss = criterion(output, labels)
            # Compute gradients
            loss.backward()
            # Perform one step of SGD optimizer
            optimizer.step()
            # Store evolution of loss
            running_loss += loss.item()
            loss_musketeer.append(loss.item())
        #print("Time this epoch:",time()-t_start)
        if e%10==0:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_dataloader)))
        correct = 0
        total = 0
        with torch.no_grad():
            for images,labels in test_dataloader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy_list.append(correct / total)
    params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(int(epochs/ratio_changes))
    np.save('musketeer'+str(eta)+'/loss_mnist_musketeer_'+params_save+'.npy',loss_cairos)
    np.save('musketeer'+str(eta)+'/accuracy_mnist_musketeer_'+params_save+'.npy',accuracy_list)
    np.save('musketeer'+str(eta)+'/g_info_'+params_save+'.npy',np.array(optimizer.param_groups[0]['g_info'].cpu()))
    return None

########## Parameter Configuration ##########
batch_size = 32
input_size = 28*28
hidden_size = 64
output_size = 10
lr = 0.01
epochs = 4
ratio_changes=0.1
#############################################


### SGD training part
for seed in range(1,11):
    print('seed=',seed)
    train_sgd(seed=seed,batch_size=batch_size,epochs=epochs,
              input_size=input_size,hidden_size=hidden_size,output_size=output_size,
              lr=lr)
    
### MUSKETEER training part with eta=1
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,batch_size=batch_size,epochs=epochs,
                    input_size=input_size,hidden_size=hidden_size,output_size=output_size,
                    lr=lr,ratio_changes=ratio_changes,eta=1)
    
### MUSKETEER training part with eta=2
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,batch_size=batch_size,epochs=epochs,
                    input_size=input_size,hidden_size=hidden_size,output_size=output_size,
                    lr=lr,ratio_changes=ratio_changes,eta=2)
    
 ### MUSKETEER training part with eta=10
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,batch_size=batch_size,epochs=epochs,
                    input_size=input_size,hidden_size=hidden_size,output_size=output_size,
                    lr=lr,ratio_changes=ratio_changes,eta=10)