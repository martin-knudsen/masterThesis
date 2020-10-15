#!/usr/bin/env python
# coding: utf-8

# Test out the parameter shift rule for a simple two mode circuit.
# 
# Import stuff

# In[1]:


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Import strawberryfields with two qumodes

# In[2]:


dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=30)


# The input is encoded to the displacement of the x-operator, which is also the output measurement. The circuit is a basic 2 mode example, with only one learnable parameter. 

# In[3]:


@qml.qnode(dev)
def variational_circuit(parameter):
    # Encode input x into quantum state
    qml.Displacement(1.0, 0.0, wires=0)
    qml.Displacement(0.5, 0.0, wires=1)
    
    # the main circuit
    qml.Beamsplitter(parameter, 0.0, wires=[0,1])    
    qml.Rotation(0.1,wires=0)
    qml.Rotation(0.2, wires=1)
    
    return [qml.expval(qml.X(0)), qml.expval(qml.X(1))]


# The function we want to approximate with the circuit is f(1,0)=1

# In[4]:


goal = 1.0


# Optimization iteration using the parameter shift rule. 

# In[5]:


lr = 0.01
costs = []
steps = 100
s = np.pi/2
parameter = 0.5
outputs =  []
parameters = []

for it in range(steps):
    # get current output of circuit and calculate square loss
    output = variational_circuit(parameter)
    outputs.append(sum(output))
    loss=(goal - sum(output)) ** 2
    costs.append(loss)  
    parameters.append(parameter)
    
    # calculate the ouput at parameter shifts using the special value pi/2 for a beamsplitter
    output_plus =  variational_circuit(parameter+s)
    output_minus = variational_circuit(parameter-s)
    
    # calculate gradient of output for both mode outputs using the parameter shift rule
    gradient_0 = 1./2.*(output_plus[0]-output_minus[0])
    gradient_1 = 1./2.*(output_plus[1]-output_minus[1])
    
    # Calculate the total gradient using the chainrule. 
    gradient = -2*(goal-sum(output))*(gradient_0+gradient_1)
    
    # update using simple gradient descent
    parameter -= gradient*lr
    


# Plot stuff

# In[6]:


fig = plt.figure()
plt.plot(range(50),costs[:50])
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel("loss",rotation='horizontal', labelpad=24, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
handles, label= plt.gca().get_legend_handles_labels()
plt.show()


# In[7]:


fig = plt.figure()
plt.plot(range(50),parameters[:50])
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel(r'$\theta$',rotation='horizontal', labelpad=14, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()


# In[8]:


fig = plt.figure()
plt.plot(range(50),outputs[:50])
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel(r'$f(\mathbf{x};\theta)$',rotation='horizontal', labelpad=34, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()

