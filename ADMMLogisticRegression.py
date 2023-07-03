#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division


# In[ ]:


import pandas as pd
import numpy as np
from mpi4py import MPI
import os
import time


# In[ ]:


dirpath = "/gpfs/projects/AMS598/Projects2022/project4/"


# In[ ]:


def chunks(files, p):
    size = int(len(files)/p)
    for i in range(0, len(files), size):
        yield files[i:i + size]


# In[ ]:


def indvar(file):
    #remove dependent variable column from matrix and add in intercept column. transforms to numpy array
    df = pd.read_csv(file).drop('y', axis=1)
    df.insert(loc=0, column='x0', value=1)
    X = df.values
    return X


# In[ ]:


def depvar(file):
    #keep only the dependent variable as a numpy array
    df = pd.read_csv(file,usecols=['y'])
    y = df['y'].ravel()[:, np.newaxis]
    return y


# In[ ]:


def logit(X):
    # function for binary response variable
    return 1 / (1 + np.exp(-X))


# In[ ]:


def h_X(beta, X):
    # returns the logit of the dot product of the weights and the features
    return logit(np.dot(X,beta))


# In[ ]:


def costfunc(beta,betabar,u,rho,X,y):
    # returns the cost function for h(x)
    m = X.shape[0]
    ss = beta-betabar+u
    cost = -(1 / m) * (np.sum(y * np.log(h_X(beta,X)) + (1 - y) * np.log(1 - h_X(beta,X))) + (rho/2) * np.dot(ss.T,ss).item())
    return cost


# In[ ]:


def gradient(beta,X,y):
    # returns the gradient of the cost function at each beta
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, h_X(beta,X) - y)


# In[ ]:


def  uVector(u,beta,betabar):
    unew = u + (beta - betabar)
    return unew


# In[63]:


def LogisticRegressionGD(u,rho,X,y,learningrate,beta=None,betabar=None):

    if type(beta) != np.ndarray:
        beta = np.zeros((X.shape[1], 1)) 
        betabar = np.zeros((X.shape[1], 1))
    else:
        beta
        betabar
        
    cost_lst = []
    lastcost = 0
    i = 0
    while True:
        gradients = gradient(beta,X,y)
        beta = beta - learningrate * gradients
        cost = costfunc(beta,betabar,u,rho,X,y)
        cost_lst.append(cost)
        i+=1
        diff = abs(cost - lastcost)
        if diff <= 1e-09:
            break
        elif i == 25000:
            break
            #print('did not converge')
        lastcost = cost
        
    #plt.plot(np.arange(1,i),cost_lst[1:], color = 'red')
    #plt.title('Cost function Graph')
    #plt.xlabel('Number of iterations')
    #plt.ylabel('Cost')
    return beta


# In[ ]:


def run():
    #set mpi variables
    comm = MPI.COMM_WORLD
    myrank = comm.rank
    p = comm.size
    
    start = time.time()
    
    if comm.rank == 0:
        allfiles = os.listdir(dirpath)
        files = ['|'.join(f) for f in list(chunks(allfiles, p)) ]
    else:
        files = None
            
    files = comm.scatter(files, root=0)
    
    files = files.split('|')
    
    X = indvar(dirpath+files[0])
    y = depvar(dirpath+files[0])
    u = np.zeros((X.shape[1], 1)) 
    beta = LogisticRegressionGD(u,1,X,y,.2)
    
    comm.Barrier()
    
    betalist = comm.gather(beta, root=0)
    
    if comm.rank == 0:
        betabar = np.mean(betalist, axis=0)
    else:
        betabar = None
    
    i = 0
        
    while True:   
        betabar = comm.bcast(betabar, root=0)
        
        conv = comm.gather(np.isclose(np.round(beta,2),np.round(betabar,2)).all(), root=0)
        
        if comm.rank == 0 and all(beta is True for beta in conv):
            break
        elif i == 25:
            break

        u = uVector(u,beta,betabar)
        
        beta = LogisticRegressionGD(u,1,X,y,.2,beta,betabar)

        comm.Barrier()

        betalist = comm.gather(beta, root=0)  
        if comm.rank == 0:
            betabar = np.mean(betalist, axis=0)
        else:
            betabar = None
        i += 1

        betabar = comm.bcast(betabar, root=0)
    end = time.time()
    
    if comm.rank == 0:
        print(betabar)
        print(end-start)
        print(i)
        np.savetxt(r"/gpfs/projects/AMS598/class2022/bdanna/project4/res.txt", betabar, delimiter=',')
    else:
        None


# In[ ]:


if __name__ == "__main__":
        run();


# In[ ]:




