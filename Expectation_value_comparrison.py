#!/usr/bin/env python
# coding: utf-8

# Comparison of expectation value as a function of shots fired with the theoretical value.
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


# In[33]:


# For different shot numbers, define a new optical quantum computer using the number of shots.
# The circuit is just a simple displacement of the ground state. 
shotss = [i for i in range(100,50000,100)]
exp_vals = []
for shots in shotss:
    dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=200, shots=shots,analytic=False)
    @qml.qnode(dev)
    def variational_circuit(x=None):
        # Encode input x into quantum state
        qml.Displacement(x, 0.0, wires=0)
        return qml.expval(qml.X(0))
     
    output = variational_circuit(x=input)
    exp_vals.append(output)

# plot
fig = plt.figure()
plt.plot(exp_vals,shotss, '.')
plt.plot([1,1],[0,50000], 'r-')
plt.yticks()
plt.xlabel("$<x>$",size=22)
plt.ylabel("shots",rotation='horizontal', labelpad=20, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
handles, label= plt.gca().get_legend_handles_labels()
plt.legend(['experimental','analytical'],prop={'size': 15},loc='upper left')
fig.savefig('expectation_val_steps.pdf'.format(shots),bbox_inches='tight')
plt.show()

