#!/usr/bin/env python
# coding: utf-8

# Scipt performing classification of the Iris dataset on a CV QNN using Penny Lane with the Pytorch interface. The neural network template is taken from the PennyLane website: https://pennylane.readthedocs.io/en/stable/_modules/pennylane/templates/layers/cv_neural_net.html#CVNeuralNetLayers

# Uses Pennylane, Sklearn, Pytorch and numpy 

# In[1]:


from pennylane.ops import Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
from pennylane.templates import broadcast
from pennylane.templates.utils import check_wires, check_number_of_layers, check_shapes
from pennylane import device, qnode, expval, X
from pennylane.init import cvqnn_layers_all
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import scale
import sklearn.decomposition
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from torch.nn.functional import softmax
cross_entropy = torch.nn.CrossEntropyLoss()


# Hyperparameters get defined here

# In[2]:


# number of layers
L=1
# number of modes
M=3
# number of beamsplitters in each interferometer
K=(M*(M-1))/2
# wire indices
wires=[i for i in range(M)]
# cutoff
cutoff_dim=4
# learning rate
lr = 0.03
# training steps
steps = 50

dev = device("strawberryfields.fock",wires=M, cutoff_dim=cutoff_dim)


# One single layer consisting of 2 interferometers, 1 squeezing, 1 displacement one Kerr nonlinearity 

# In[3]:


def cv_neural_net_layer(
    theta_1, 
    phi_1, 
    varphi_1, 
    r, 
    phi_r, 
    theta_2, 
    phi_2, 
    varphi_2, 
    a, 
    phi_a, 
    k, 
    wires):
    
    Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, wires=wires)

    broadcast(unitary=Squeezing, pattern="single", wires=wires, parameters=list(zip(r, phi_r)))

    Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, wires=wires)

    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(a, phi_a)))

    broadcast(unitary=Kerr, pattern="single", wires=wires, parameters=k)


# Several quantum Neural network layers

# In[4]:


def CVNeuralNetLayersHomeMade(
    theta_1, 
    phi_1, 
    varphi_1, 
    r, 
    phi_r, 
    theta_2, 
    phi_2, 
    varphi_2, 
    a, 
    phi_a, 
    k, 
    wires):

    #############
    # Input checks
    wires = check_wires(wires)

    n_wires = len(wires)
    n_if = n_wires * (n_wires - 1) // 2
    weights_list = [theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k]
    repeat = check_number_of_layers(weights_list)

    expected_shapes = [
        (repeat, n_if),
        (repeat, n_if),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_if),
        (repeat, n_if),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
        (repeat, n_wires),
    ]
    check_shapes(weights_list, expected_shapes, msg="wrong shape of weight input(s) detected")

    ###############

    for l in range(repeat):
        cv_neural_net_layer(
            theta_1=theta_1[l],
            phi_1=phi_1[l],
            varphi_1=varphi_1[l],
            r=r[l],
            phi_r=phi_r[l],
            theta_2=theta_2[l],
            phi_2=phi_2[l],
            varphi_2=varphi_2[l],
            a=a[l],
            phi_a=phi_a[l],
            k=k[l],
            wires=wires)


# Quantum circuit performing first an encoding of the input float values, A feed forward through the NN layers and ending with an expectation value of X of all wires. 

# In[5]:


@qnode(dev, interface='torch')
def quantum_circ(params, wires=None, x=None, phi_x=None):
    
    # Encode input x into quantum state
    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(x, phi_x)))
    
    CVNeuralNetLayersHomeMade(
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
        params[10],
        wires)
    
    return [expval(X(i)) for i in range(M)]


# Init values at random using Pennylane.init and convert to Pytorch Tensor

# In[6]:


inits = torch.tensor(cvqnn_layers_all(L, M, seed=0))
print(inits)


# Test out the quanum circuit with the initialized weights and some random inputs

# In[7]:


print(quantum_circ(inits, wires=wires, x=[0.1, 0.3, 0.1], phi_x=[0, 0, 0]))


# Preprocess and plot the Iris dataset using first PCA with two components for plotting.

# In[8]:


iris = load_iris()
data = iris.data
target = iris.target
zeros = (target==0)
ones = (target==1)
twos = (target==2)
target_names= ['Setosa', 'Versicolor', 'Virginica']
n_features=2
pca = sklearn.decomposition.PCA(n_components=n_features)
pca.fit(data)
data = pca.transform(data)
plt.scatter(data[zeros,0],data[zeros,1],c='r')
plt.scatter(data[ones,0],data[ones,1],c='g')
plt.scatter(data[twos,0],data[twos,1],c='b')
plt.yticks()
plt.xlabel("PC1",size=22)
plt.ylabel("PC2",rotation='horizontal', labelpad=9, size=22)
plt.title('')
plt.xlim(-4,6.3)
plt.ylim(-2,1.6)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.legend(target_names,prop={'size': 16},loc='lower right')
plt.show()


# Preprocess and plot the Iris dataset using first PCA with three components for plotting.

# In[9]:


iris = load_iris()
data = iris.data
target = iris.target

# Use PCA to decrease the dimension of the Input to the quantum circuit
n_features=3
pca = sklearn.decomposition.PCA(n_components=n_features)
pca.fit(data)
data = pca.transform(data)

# dividing data and target into train and test data and scale input to be in the interval [-1,1]
data_train, data_test, target_train, target_test = train_test_split(data, target, random_state = 0) 
data_train = scale(data_train)
n = len(data_train)

# Convert the dataset to the Pytorch tensor datatype 
data_train = torch.tensor(data_train) 
data_test= torch.tensor(data_test) 
target_train= torch.tensor(target_train) 
target_test= torch.tensor(target_test) 


# Create wrapper function to perform the feedforward of all data points

# In[10]:


def quantum_circ_wrapper(params, data):
    phi_x = torch.zeros([3])
    n = data.size()[0]
    preds = torch.zeros([n,3], requires_grad=False)    
    for i in range(n):
        preds[i] = quantum_circ(params, wires=wires, x=data[i], phi_x=phi_x)
        
    return preds


# Define loss as crossentropy using the standard class from Pytorch

# In[11]:


def accuracy(results, targets):
    size=results.size()[0]
    preds_softmax = softmax(results,dim=1)
    choice = torch.argmax(preds_softmax, dim=1)
    corrects = [targets[i].item()==choice[i].item() for i in range(size)]
    correct = sum(corrects)
    normalized = correct/size
    return normalized


# Step function and optimizer gets defined

# In[13]:


# make the parameters become a Variable for Pytorch to optimize them
var = Variable(inits, requires_grad=True)
opt = torch.optim.Adam([var], lr = lr)

# closure function defining a feed forward and a backward propagation. 
# The outputs and loss are saved in the function field variables for querying. 
def closure():
    opt.zero_grad()
    closure.output_train = quantum_circ_wrapper(var, data_train)
    loss = cross_entropy(closure.output_train, target_train)
    loss.retain_grad()
    closure.loss = loss.item()
    loss.backward()
    return loss


# Perform training and save accuracy and loss for both training and testing sets

# In[14]:


# training by alternatingly feeding forward and backpropagating
loss_train_list = []
loss_test_list = []
acc_train_list = []
acc_test_list = []
for it in range(steps):         
    
    # before stepping use var's toget test loss and accuracy
    output_test = quantum_circ_wrapper(var, data_test)
    loss_test = cross_entropy(output_test, target_test).item()
    acc_test = accuracy(output_test,target_test)
    opt.step(closure)
    
    # gather the loss and outputs from the closure function 
    output_train = closure.output_train
    loss_train = closure.loss    
    acc_train = accuracy(output_train,target_train)
    
    loss_train_list.append(loss_train)
    loss_test_list.append(loss_test)
    acc_train_list.append(acc_train)
    acc_test_list.append(acc_test)
    
    print("Iter: {:5d} | Cost: {} ".format(it, loss_train))    


# load and plot the training data

# In[18]:


plt.plot(range(len(acc_train_list)), acc_train_list, label='accuracy')
plt.plot(range(len(acc_train_list)), loss_train_list, label='loss')
plt.xlabel('step', size=20)
plt.title('Training', size=20)
handles, label= plt.gca().get_legend_handles_labels()
plt.legend([handles[0],handles[1]],['accuracy','loss'],prop={'size': 16},loc='upper right')
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()

plt.plot(range(len(acc_train_list)), acc_test_list, label='accuracy')
plt.plot(range(len(acc_train_list)), loss_test_list, label='loss')
plt.xlabel('step', size=20)
plt.title('Testing', size=20)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
handles, label= plt.gca().get_legend_handles_labels()
plt.legend([handles[0],handles[1]],['accuracy','loss'],prop={'size': 16},loc='upper right')

plt.show()

