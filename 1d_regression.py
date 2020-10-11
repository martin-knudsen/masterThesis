#!/usr/bin/env python
# coding: utf-8

# This script follows the tutorial from Pennylane https://pennylane.ai/qml/demos/quantum_neural_net.html about using a variational circuit for 1D function regression. 
# 
# Import stuff

# In[29]:


import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Import strawberryfields with one qumode

# In[30]:


dev = qml.device("strawberryfields.fock", wires=1, cutoff_dim=10)


# Define one fully connected NN layer with bias as rotation, squeezing, rotation and displacement. The nonliniearity is made by the Kerr gate. All operations happening in the circuit are made to "qml". 

# In[31]:


def layer(v):
    # Matrix multiplication of input layer
    qml.Rotation(v[0], wires=0)
    qml.Squeezing(v[1], 0.0, wires=0)
    qml.Rotation(v[2], wires=0)

    # Bias
    qml.Displacement(v[3], 0.0, wires=0)

    # Element-wise nonlinear transformation
    qml.Kerr(v[4], wires=0)


# The input is encoded to the displacement of the x-operator, which is also the output measurement. We perform all the layers and then return the expectation value of the first component of X. 

# In[32]:


@qml.qnode(dev)
def quantum_neural_net(var, x=None):
    # Encode input x into quantum state
    qml.Displacement(x, 0.0, wires=0)

    # "layer" subcircuits
    for v in var:
        layer(v)

    return qml.expval(qml.X(0))


# Loss function is taken as the average of the square euclidian norm

# In[33]:


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


# The cost function calculates the outcome of the circuit for each datapoint, i.e. one feed forward and a loss function to the total result. 

# In[34]:


def cost(var, features, labels):
    cost.preds = np.array([quantum_neural_net(var, x=x) for x in features])
    cost.loss=square_loss(labels, cost.preds)
    return cost.loss


# create the data to fit. Here we use a gaussian function with some random noise added.

# In[35]:


X = np.linspace(-1,1,40)
def gaussian(x, mu=0, sig=0.3):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
Y = gaussian(X)
noise = np.random.randn(40)*0.04-0.01
Y = Y+ noise


# which looks like this

# In[36]:


fig = plt.figure()
plt.scatter(X, Y)
plt.yticks()
plt.xlabel("x",size=22)
plt.ylabel("f(x)",rotation='horizontal', labelpad=15, size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()


# We sample the weights of the NN with random samples from a normal distribution. We use 4 layers. 

# In[37]:


np.random.seed(0)
num_layers = 4
var_init = 0.05*3 * np.random.randn(num_layers, 5)
print(var_init)


# We optimize the weights using the cost function and the Adamoptimizer and 1000 steps. The optimizer gets as input the cost function as a lambda function plus the previous values of the variables. 

# In[38]:


opt = AdamOptimizer(0.02, beta1=0.9, beta2=0.999)
var = var_init
costs = []
for it in range(1000):
    
    # step using the pennylane numpy. In order to save execution time, the cost and predictions 
    # are directly saved in the wrapper cost funtion, so they can be accessed. 
    var = opt.step(lambda v: cost(v, X, Y), var)
    costLoss = cost.loss._value
    predictions = cost.preds._value
    costs.append(costLoss)

# plot
fig = plt.figure()
plt.scatter(X, Y,label='Data')
plt.plot(X, predictions, color="green",label='QNN')
plt.yticks()
plt.xlabel("x",size=22)
plt.ylabel("f(x)",rotation='horizontal', labelpad=15, size=22)
plt.title('Step: {}, MSE: {:0.7f}'.format(it, costLoss), size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
handles, label= plt.gca().get_legend_handles_labels()
plt.legend([handles[1],handles[0]],['Data','QNN'],prop={'size': 16},loc='upper left')
plt.show()


# Plot losses

# In[40]:


fig = plt.figure()
plt.plot(range(len(costs)),costs)
plt.yticks()
plt.xlabel("step",size=22)
plt.ylabel("MSE",rotation='horizontal', labelpad=15, size=22)
plt.title('MSE loss', size=22)
plt.tick_params(axis="both", which="major",labelsize=22)
plt.tick_params(axis="both", which="minor",labelsize=22)
plt.show()

