#!/usr/bin/env python
# coding: utf-8

# Script for training a CV neural network to recreate the 2D Rosenbrock function. The neural network template is taken from the PennyLane website: https://pennylane.readthedocs.io/en/stable/_modules/pennylane/templates/layers/cv_neural_net.html#CVNeuralNetLayers
# 
# Import stuff

# In[48]:


from pennylane.ops import Squeezing, Displacement, Kerr
from pennylane.templates.subroutines import Interferometer
from pennylane.templates import broadcast
from pennylane import numpy as np
from pennylane.templates.utils import check_wires, check_number_of_layers, check_shapes
from pennylane import device, qnode, expval, X
from pennylane.init import (cvqnn_layers_all)
from pennylane.optimize import AdamOptimizer
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot
import seaborn as sns
sns.set()
import numpy as np


# Syperparameters

# In[49]:


# number of layers
L=4
# number of modes
M=2
# number of beamsplitters in each interferometer
K=(M*(M-1))/2
# wire indices
wires=[i for i in range(M)]

dev = device("strawberryfields.fock",wires=M, cutoff_dim=10)


# One single layer (taken from PennyLane templates, see top)

# In[50]:


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
    
    # unitary transformation
    Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, wires=wires)
    
    # scaling
    broadcast(unitary=Squeezing, pattern="single", wires=wires, parameters=list(zip(r, phi_r)))
    
    # unitary transformation
    Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, wires=wires)
    
    # bias
    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(a, phi_a)))
    
    # non-linearity
    broadcast(unitary=Kerr, pattern="single", wires=wires, parameters=k)


# Multiple layers (taken from PennyLane templates, see top)

# In[51]:


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

    ############### Do for all layers

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
            wires=wires,
        )


# Init values using the PennyLane function for a CV netork. 

# In[52]:


inits = cvqnn_layers_all(L, M, seed=0)
print(inits)


# Wrapper quantum circuit function containing the embedding of the function point and the neural network

# In[53]:


@qnode(dev)
def quantum_circ(params, x=None):
    # Encode input x into quantum state
    Displacement(x[0], 0.0, wires=0)
    Displacement(x[1], 0.0, wires=1)
    
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
        [0,1])
    
    return expval(X(0))


# Tryout the quantum circuit

# In[54]:


print(quantum_circ(inits, x=[0.1, 0.3]))
print(quantum_circ(inits, x=[0.2, 0.2]))


# Plot the normalized Rosenbrock function

# In[55]:


XX = np.linspace(-2, 2., 30)   #Could use linspace instead if dividing
YY = np.linspace(-2, 4., 30)   #evenly instead of stepping...

#Create the mesh grid(s) for all X/Y combos.
XX, YY = np.meshgrid(XX, YY)

#Rosenbrock function w/ two parameters using numpy Arrays and normalized
ZZ = (1.-XX)**2 + 100.*(YY-XX*XX)**2
ZZ = ZZ/np.max(ZZ)

# plot
ax = fig.gca(projection='3d')
ax.view_init(30, 35)
ax.dist = 13
surf = ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, 
         cmap='coolwarm', edgecolor='none',antialiased=True)  #Try coolwarm vs jet
ax.set_xlabel('x', rotation=0, labelpad=10, size=22)
ax.set_ylabel('y', rotation=0, labelpad=10, size=22)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2,4])
ax.set_zticks([0, 1])
ax.set_xlim([-2,2])
ax.set_ylim([-2,4])
ax.set_zlim([0,1])
ax.tick_params(axis="both", which="major",labelsize=16)
ax.tick_params(axis="both", which="minor",labelsize=16)
ax.set_zlabel('f(x,y)', size=10, rotation='horizontal',rotation_mode='anchor')
plot.show()


# Define square loss between a list of labels and predictions

# In[56]:


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


# Cost function wrapper creates output from circuit and calculates loss. Data saved for faster execution.

# In[57]:


def cost(params, features, labels):
    cost.preds = [quantum_circ(params, x=x) for x in features]
    cost.loss=square_loss(labels, cost.preds)
    return cost.loss


# Define the the points at a coarser grid for good execution time. Then use Adamoptimizer to train the network.

# In[58]:


XX = np.linspace(-2, 2., 5)   #Could use linspace instead if dividing
YY = np.linspace(-2, 4., 5)   #evenly instead of stepping...

#Create the mesh grid(s) for all X/Y combos.
XX, YY = np.meshgrid(XX, YY)

#Rosenbrock function w/ two parameters using numpy Arrays
ZZ = (1.-XX)**2 + 100.*(YY-XX*XX)**2
ZZ = ZZ/np.max(ZZ)

# final inputs and labels
inputs = list(zip(XX.flatten(),YY.flatten()))
labels = ZZ.flatten()

# sets initial variable value and optimizer
var = inits
opt = AdamOptimizer(0.01, beta1=0.9, beta2=0.999)

losses = []
for it in range(500):
    
    # update variable with optimizer and get loss
    var = opt.step(lambda v: cost(v, inputs,labels), var)
    losses.append(cost.loss._value)
    
# get the final predictions
preds = np.array([pred._value for pred in cost.preds])
ZZ_NN = preds.reshape(ZZ.shape)

# plot
ax = fig.gca(projection='3d')
ax.view_init(30, 35)
ax.dist = 13
surf = ax.plot_surface(XX, YY, ZZ_NN, rstride=1, cstride=1, 
         cmap='coolwarm', edgecolor=None, antialiased=True)  #Try coolwarm vs jet
ax.set_xlabel('x', rotation=0, labelpad=10, size=22)
ax.set_ylabel('y', rotation=0, labelpad=10, size=22)
ax.set_zlabel('f(x)', rotation=0, labelpad=10, size=22)
ax.set_xticks([-2, 0, 2])
ax.set_yticks([-2, 0, 2,4])
ax.set_zticks([-1,0, 1])
ax.set_xlim([-2,2])
ax.set_ylim([-2,4])
ax.tick_params(axis="both", which="major",labelsize=16)
ax.tick_params(axis="both", which="minor",labelsize=16)
plot.show()
    


# In[59]:


# plot the MSE loss 
plot.plot(range(len(losses)),losses)
plot.yticks()
plot.xlabel("step",size=22)
plot.ylabel("MSE",rotation='horizontal', labelpad=15, size=22)
plot.title('MSE loss', size=22)
plot.tick_params(axis="both", which="major",labelsize=22)
plot.tick_params(axis="both", which="minor",labelsize=22)
plot.show()


# ##### 
