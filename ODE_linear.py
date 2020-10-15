#!/usr/bin/env python
# coding: utf-8

# Scipt performing classification of the Iris dataset on a CV QNN using Penny Lane with the Pytorch interface. The neural network template is taken from the PennyLane website: https://pennylane.readthedocs.io/en/stable/_modules/pennylane/templates/layers/cv_neural_net.html#CVNeuralNetLayers

# Uses Pennylane, Sklearn, Pytorch and numpy 

# In[12]:


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
from torch.autograd import Variable, grad
import numpy as np
import seaborn
import matplotlib.pyplot as plt


# Hyperparameters get defined here

# In[13]:


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
steps = 100

dev = device("strawberryfields.fock",wires=M, cutoff_dim=cutoff_dim)


# One single layer consisting of 2 interferometers, 1 squeezing, 1 displacement one Kerr nonlinearity 

# In[14]:


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


# Several layers of CV neural network from PennyLane

# In[15]:


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

# In[16]:


@qnode(dev, interface='torch')
def quantum_circ(params,x, wires=None, phi_x=None):
    
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


# Init parameters at random using Pennylane.init and convert to Pytorch Tensor

# In[17]:


inits = torch.tensor(cvqnn_layers_all(L, M, seed=0), dtype=torch.float64)


# Test out the quantum circuit with the initialized weights and some random inputs

# In[18]:


print(quantum_circ(inits, [0.1, 0.3, 0.1], wires=wires, phi_x=[0, 0, 0]))


# Creating the ODE dataset following https://towardsdatascience.com/how-to-solve-an-ode-with-a-neural-network-917d11918932
# 
# We solve the following ODE y’=-2xy and y(0)=1 on the interval [-1,1]

# In[19]:


inputs = np.linspace(-1., 1., num=100)
fun_der = lambda x,y: -2.*x*y
fun_theo = lambda x: np.exp(-x**2)
y0 = 1
fx = fun_theo(inputs)

# the input data in a torch tensor
x_data = Variable(torch.linspace(-1.0,1.0,20,dtype=torch.float64),requires_grad=True)
x_data = x_data.view(20,1)


# Wrapper function that performs a hybrid computation where the QNN is sandwiched between classical layers. 
# 
# The layers are: Linear layer -> QNN -> linear layer -> tangens activation function. 
# 
# Th linear layers serve to send the input from 1 neuron to many and go back from many to 1 as an autoencoder. They do not contain weights or bias that can be optimized. 

# In[20]:


def sigmoid_wrapper(x, params):
    phi_x = torch.zeros(M)
    wires=[i for i in range(M)]
    weight1 = torch.tensor([[1.],[1.],[1.]],dtype=torch.float64)
    weight2 = torch.tensor([[1.,1.,1.]],dtype=torch.float64)
    x_array=torch.nn.functional.linear(x, weight1, bias=None)
    circ_output = quantum_circ(params,x_array, wires=wires, phi_x=phi_x)
    summ=torch.nn.functional.linear(circ_output, weight2, bias=None)
    classic_calc = torch.sigmoid(summ)
    return classic_calc


# Define loss by minimizing f(x,y) = (y'+2yx)^2+(y(0)-1)^2, so as a sum of the deviation from the ODE and the initial condition.
# 
# y' is found by differentiating the the whole NN using pytorch.autograd's grad function. 

# In[21]:


def loss_fun(inputs,var):
    n = inputs.size()[0]
    loss = torch.tensor(0.0, requires_grad=True, dtype=torch.float64)    
    loss_fun.outputs = torch.zeros(inputs.size())
    
    # for each input point get the output of the NN, the grad of the point and add the ODE loss part of that point
    for i in range(n):
        x = Variable(inputs[i], requires_grad=True)
        output = sigmoid_wrapper(x, var)
        loss_fun.outputs[i] = output
        dydx, = torch.autograd.grad(output, x, grad_outputs=output.data.new(output.shape).fill_(1),
        create_graph=True, retain_graph=True)
        eq = dydx + 2.* inputs[i] * output # y' = - 2x*y
        loss = loss.add(torch.mean(eq**2))
    
    # add the initial condition point
    ic = sigmoid_wrapper(torch.tensor([0.],dtype=torch.float64), var) - 1.    # y(x=0) = 1
    loss = loss.add(ic**2)
    return loss


# Perform training and save accuracy and loss for both training and testing sets

# In[22]:


var = Variable(inits, requires_grad=True)
opt = torch.optim.Adam([var], lr = 0.1)
for it in range(steps):         
    
    opt.zero_grad()
    loss = loss_fun(x_data, var)
    
    loss.backward()
    opt.step()


    print('it {}, loss {}'.format(it, loss.item()))
    
    # Plot Results
    fig=plt.figure()
    plt.plot(x_data.data.numpy(), np.exp(-x_data.data.numpy()**2), label='exact')
    plt.plot(x_data.data.numpy(), loss_fun.outputs.data.numpy(), label='approx')
    plt.legend()
    plt.show()

