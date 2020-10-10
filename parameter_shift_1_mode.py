#!/usr/bin/env python
# coding: utf-8

# Test out the parameter shift rule for a simple single mode circuit.
# 
# Import stuff

# In[8]:


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Import strawberryfields Fock backend with one qumode

# In[9]:


shots = 10000
dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=200, shots=shots,analytic=False)


# The input is encoded to the displacement of the x-operator, which is also the output measurement. After encoding the only other gate is another displacement gate, however, it has a learnable parameter. 

# In[10]:


@qml.qnode(dev)
def variational_circuit(parameter):
    # Encode input x into quantum state
    qml.Displacement(0.0, 0.0, wires=0)    
    qml.Displacement(parameter, 0.0, wires=0)    
    return qml.expval(qml.X(0))


# The circuit will solve the function f(0)=1. 

# In[11]:


input = 0
goal = 1.0


# Optimization iteration using the parameter shift rule. 

# In[16]:


lr = 0.01
costs = []
steps = 100
np.random.seed(0)
s = 1
parameter = -0.1

for it in range(steps):
    # output with current parameter
    output = variational_circuit(parameter)
    loss=(goal - output) ** 2
    costs.append(loss)  

    # output at equal shifts in the parameter
    output_plus =  variational_circuit(parameter+s)
    output_minus = variational_circuit(parameter-s)

    # gradient of circuit using parameter shift rule 
    output_gradient = 1./(2.*s)*(output_plus-output_minus)
    
    # calculate the gradient of the loss function with respect to the parameter using the chain rule.
    gradient = -2*(goal-output)*output_gradient
    
    # update parameter with simple gradient descent 
    parameter -= gradient*lr

# plot
plt.plot(range(steps), costs)
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel("cost",rotation='horizontal', labelpad=20, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()

