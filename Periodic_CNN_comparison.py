#!/usr/bin/env python
# coding: utf-8

# Code for implementing a periodic convolutional neural network on a CV quantum computer 

# In[2]:


from pennylane.templates.decorator import template
from pennylane.ops import Squeezing, Displacement, Kerr, Interferometer, Rotation
#from pennylane.templates.subroutines import Interferometer
from pennylane.templates import broadcast
from pennylane import numpy as np
from pennylane.templates.utils import check_wires, check_number_of_layers, check_shapes, check_type
from pennylane import device, qnode, expval, X, probs, P
from pennylane.init import (cvqnn_layers_all)
from pennylane.optimize import AdamOptimizer
from scipy.linalg import circulant
import seaborn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot


# hyperparameters

# In[3]:


# number of layers
L=4
# number of modes
M=10
# number of beamsplitters in each interferometer
K=(M*(M-1))/2
# wire indices
wires=[i for i in range(M)]

dev = device("strawberryfields.fock",wires=M, cutoff_dim=2)


# One single layer using an altered neural network layer without bias and non-linearity as well as an Interferometer that implements a unitary transformation. 

# In[4]:


@qnode(dev)
def quantum_circ1(x=None, phi_x=None, U1=None, U2=None, r=None, phi_r=None, phi_rot=None):
    
    M = len(x)
    wires = [wire for wire in range(M)]
    
    broadcast(unitary=Displacement, pattern="single", wires=wires, parameters=list(zip(x, phi_x)))
    
    Interferometer(U1, wires=wires)
    
    broadcast(unitary=Squeezing, pattern="single", wires=wires, parameters=list(zip(r, phi_r)))
    
    broadcast(unitary=Rotation, pattern="single", wires=wires, parameters=list(phi_rot))
    
    Interferometer(U2, wires=wires)
    
    return [expval(X(i)) for i in range(M)]


# Test out difference between CV neural network convolution and the theoretical convolution value. 

# In[5]:


F = np.fft.fft(np.eye(M))/(np.sqrt(M))
F_H = F.conj().T

# random values to transform
x=np.random.random(M)
phi_x = np.zeros(M)   
phi_r = np.zeros(M)    
np.random.seed(1)

# initialize some random vals for the periodic convolution
init = np.random.uniform(low=-1, high=1, size=(2))

c = np.zeros(M)
c[0]=init[0]
c[1]=init[1]

# get the fourier transformed values and seperate argument and absolute value
complex_vals = np.fft.fft(c)
scale_arg = np.angle(complex_vals)
scale_abs = np.absolute(complex_vals)

# get the required squeeze scaling argument
r1 = -np.log(scale_abs)

# perform a neural network feed forward using the transformed Fourier matrix, the required phase shift using a rotation gate (phase gate),
# the required squeeze scaling and the Fourier matrix

circ_res = quantum_circ1(x=x, phi_x=phi_x, U1=F_H, U2=F, r=r1, phi_r=phi_r, phi_rot=scale_arg)

# theoretical transformation
mat=F@np.diag(scale_abs*np.exp(1J*scale_arg))@F_H

# applied theoretical result to the input
theo_res = np.real(mat@x)

# difference int the theoretical transform and the actual result of the circuit
print('theo')
print(theo_res)
print('actual')
print(circ_res)

