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
import torch
from torch.autograd import Variable, grad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.integrate import odeint


# Hyperparameters get defined here

# In[2]:


# number of layers
L=1
# number of modes
M=2
# number of beamsplitters in each interferometer
K=int((M*(M-1))/2)
# wire indices
wires=[i for i in range(M)]
# cutoff
cutoff_dim=4
# learning rate
lr = 0.02
# training steps
steps = 1000
# intervals for the parmeters because they have different sizes
sizes = [K,K,M,M,M,K,K,M,M,M,M]
sizes = np.cumsum(sizes)

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


# Several layers

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


# Quantum circuit performing first an encoding of the input float values, A feed forward through the NN layers and ending with an expectation value of X of all wires. The sizes structure is to be able to keep data in a torch tensor. 

# In[5]:


@qnode(dev, interface='torch')
def quantum_circ(params,x, wires=None, phi_x=None):
    
    # Encode input x into quantum state
    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(x, phi_x)))
    CVNeuralNetLayersHomeMade(
        params[:][:,0:sizes[0]],
        params[:][:,sizes[0]:sizes[1]],
        params[:][:,sizes[1]:sizes[2]],
        params[:][:,sizes[2]:sizes[3]],
        params[:][:,sizes[3]:sizes[4]],
        params[:][:,sizes[4]:sizes[5]],
        params[:][:,sizes[5]:sizes[6]],
        params[:][:,sizes[6]:sizes[7]],
        params[:][:,sizes[7]:sizes[8]],
        params[:][:,sizes[8]:sizes[9]],
        params[:][:,sizes[9]:sizes[10]],
        wires)
    
    return [expval(X(i)) for i in range(M)]


# Init parameters at random using Pennylane.init and convert to Pytorch Tensor

# In[6]:


inits = cvqnn_layers_all(L, M)
inits = [torch.tensor(i) for i in inits]
inits = torch.cat(inits, dim=1)


# Test out the quanum circuit with the initialized weights and some random inputs

# In[7]:


print(quantum_circ(inits, torch.rand(M), wires=wires, phi_x=torch.zeros(M)))


# Creating the ODE dataset following https://towardsdatascience.com/how-to-solve-an-ode-with-a-neural-network-917d11918932
# 
# We solve the following ODE y’=x^2+y^2-1 and y(0)=0 on the interval [-1.5,1.5] from Calculus Concepts & Contexts 3, James Stewart p. 506

# In[8]:


inputs = np.linspace(-1.5, 1.5, num=100)

def fun_der(y,x): 
    return x**2+y**2-1

y0 = 0.0

# the input data in a torch tensor
x_data = Variable(torch.linspace(-1.5,1.5,20,dtype=torch.float64),requires_grad=True)
x_data = x_data.view(20,1)

# theoretical solution using the odeint function from scipy. Start from 0 and go in each direction ao the initial value is y(0)=0
xpos = np.linspace(0, 1.5, num=50)
ypos = odeint(fun_der, y0, xpos)
ypos = np.array(ypos).flatten()
xneg = np.linspace(0, -1.5, num=50)
yneg = odeint(fun_der, y0, xneg)
yneg = np.array(yneg).flatten()

xs = np.concatenate((xneg[::-1],xpos))
ys = np.concatenate((yneg[::-1],ypos))


# Wrapper function that performs a hybrid computation where the QNN is sandwiched between classical layers. 
# 
# The layers are: Linear layer -> QNN -> linear layer
# 
# Th linear layers serve to send the input from 1 neuron to many and go back from many to 1 as an autoencoder. They do not contain weights or bias that can be optimized.
# 
# The only classical postprocessing is a summation of the expectation values

# In[9]:


def sum_wrapper(x, params):
    phi_x = torch.zeros(M)
    wires=[i for i in range(M)]
    weight1 = torch.ones((M,1),dtype=torch.float64)
    weight2 = torch.ones((1,M),dtype=torch.float64)
    x_array=torch.nn.functional.linear(x, weight1, bias=None)
    circ_output = quantum_circ(params,x_array, wires=wires, phi_x=phi_x)
    summ=torch.nn.functional.linear(circ_output, weight2, bias=None)
    return summ


# Define loss by minimizing f(x,y) = (y'+2yx)^2+(y(0)-1)^2, so as a sum of the deviation from the ODE and the initial condition.
# 
# y' is found by differentiating the the whole NN using pytorch.autograd's grad function. 

# In[10]:


def loss_fun(inputs,var):
    n = inputs.size()[0]
    loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)    
    loss_fun.outputs = torch.zeros(inputs.size())
    
    # for each input point get the output of the NN, the grad of the point and add the ODE loss part of that point
    
    for i in range(n):
        x = Variable(inputs[i], requires_grad=True)
        output = sum_wrapper(x, var)
        loss_fun.outputs[i] = output
        dydx, = torch.autograd.grad(output, x, grad_outputs=output.data.new(output.shape).fill_(1),
        create_graph=True, retain_graph=True)
        eq = dydx-inputs[i]**2-output**2+1 # y' = x^2+y^2-1
        loss = loss.add(torch.mean(eq**2))
    
    # add the initial condition point
    ic = sum_wrapper(torch.tensor([0.],dtype=torch.float64), var)     # y(x=0) = 0.0
    loss = loss.add(ic**2)
    return loss


# Perform training and save accuracy and loss for both training and testing sets

# In[11]:


var = Variable(inits, requires_grad=True)
opt = torch.optim.Adam([var], lr=lr, eps=0.001)
losses = []
varss = []
for it in range(steps):
    varss.append(var.detach().clone())
    # Feedforward and backpropagation + step
    
    opt.zero_grad()
    loss = loss_fun(x_data, var)
    losses.append(loss.item())
    loss.backward()
    opt.step()    

    # Plot Results     
    plt.plot(xs, ys, label='classic')
    plt.plot(x_data.data.numpy(), loss_fun.outputs.data.numpy(), label='QNN')
    plt.yticks()
    plt.xlabel("x",size=22)
    plt.ylabel("y(x)",rotation='horizontal', labelpad=20, size=22)
    plt.title('Step: {}, loss: {:0.7f}'.format(it, loss.item()), size=22)
    plt.tick_params(axis="both", which="major",labelsize=22)
    plt.tick_params(axis="both", which="minor",labelsize=22)
    plt.legend(prop={'size': 16},loc='upper right')
    plt.show()


# In[14]:


plt.plot(range(len(losses)),losses)
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel("loss",rotation='horizontal', labelpad=15, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()

