#!/usr/bin/env python
# coding: utf-8

# Output of circuit in different bins according to their frequency for 10000 shots.
# 
# Import stuff

# In[2]:


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


input = 0.5


# In[106]:


shots = 10000
dev = qml.device("default.gaussian", wires=1, shots=shots,analytic=False)

# circuit is a simple displacement of the ground state and returns a sample of size shots
@qml.qnode(dev)
def variational_circuit(x=None):
    qml.Displacement(x, 0.0, wires=0)
    return qml.sample(qml.X(0))

output = variational_circuit(x=input)

# calculate expectation value  using the Berry-Essen theorem
ev1 = np.mean(output)
var = np.var(output)
ev = np.random.normal(ev1, np.sqrt(var / shots))

# plot
min_ = min(output)
max_ = max(output)
bins = np.linspace(min_,max_,100)
fig = plt.figure()
plt.hist(output,bins)
plt.plot([ev,ev],[0,350],'r-')
plt.yticks()
plt.xlabel("$x$",size=22)
plt.ylabel("f",rotation='horizontal', labelpad=15, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.legend(['expectation value','data'],prop={'size': 12},loc='upper right')
fig.savefig('expectation_val_10000_shots.pdf'.format(shots),bbox_inches='tight')
plt.show()

