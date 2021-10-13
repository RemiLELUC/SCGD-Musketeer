import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
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
        
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward_pass(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_sgd(seed,epochs,lr):
    # set seed for reproducibility
    set_seed(seed)
    # Build Neural Network
    model = Net()
    # Put on GPU
    model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer SGD
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    # Initialize loss evolution
    loss_sgd= []
    accuracy_list = []
    # Main training loop
    for e in range(epochs):
        running_loss = 0
        t_start = time()
        for data in train_dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            # Set gradients to 0
            optimizer.zero_grad()
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
            for data in test_dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        #print('Accuracy:',correct/total)
        accuracy_list.append(correct / total)
    params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(epochs)
    np.save('sgd/loss_cifar10_sgd_'+params_save+'.npy',loss_sgd)
    np.save('sgd/accuracy_cifar10_sgd_'+params_save+'.npy',accuracy_list)
    return None
        
def train_musketeer(seed,epochs,lr,ratio_changes,eta):
    # set seed for reproducibility
    set_seed(seed)
    # Build Neural Network
    model = Net()
    # Put on GPU
    model.to(device)
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer SPARTACOS # exploration size
    d = sum(p.numel() for p in model.parameters())
    T = int(math.sqrt(d))
    optimizer = Musketeer(params=list(model.parameters()),T=T,lr=lr,
                       ratio_changes=ratio_changes,eta=eta)
    # Initialize loss evolution
    loss_musketeer = []
    accuracy_list = []
    # Main training loop
    for e in range(int(epochs/ratio_changes)):
        running_loss = 0
        t_start = time()
        for data in train_dataloader:
            # Get images,labels of current batch
            images, labels = data[0].to(device), data[1].to(device)
            # Set gradients to 0
            optimizer.zero_grad()
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
            for data in test_dataloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model.forward_pass(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        #print('Accuracy:',correct/total)
        accuracy_list.append(correct / total)
    params_save = 'seed'+str(seed)+'_lr'+str(lr)+'_epochs'+str(int(epochs/ratio_changes))+'_eta'+str(eta)
    np.save('musketeer'+str(eta)+'/loss_cifar10_musketeer_'+params_save+'.npy',loss_cairos)
    np.save('musketeer'+str(eta)+'/accuracy_cifar10_musketeer_'+params_save+'.npy',accuracy_list)
    np.save('musketeer'+str(eta)+'/g_info_'+params_save+'.npy',np.array(optimizer.param_groups[0]['g_info'].cpu()))
    return None

########## Parameter Configuration ##########
epochs= 5
lr = 0.01
ratio_changes=0.1
#############################################


### SGD training part
for seed in range(1,11):
    print('seed=',seed)
    train_sgd(seed=seed,epochs=epochs,lr=lr)
    
### MUSKETEER training part with eta=1
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,epochs=epochs,
                    lr=lr,ratio_changes=ratio_changes,eta=1)
    
### MUSKETEER training part with eta=2
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,epochs=epochs,
                    lr=lr,ratio_changes=ratio_changes,eta=2)
    
 ### MUSKETEER training part with eta=10
for seed in range(1,11):
    print('seed:',seed)
    train_musketeer(seed=seed,epochs=epochs,
                    lr=lr,ratio_changes=ratio_changes,eta=10)