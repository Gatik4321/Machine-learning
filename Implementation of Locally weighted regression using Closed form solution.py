#!/usr/bin/env python
# coding: utf-8

# # Loading Data

# In[2]:


# #Implementation of Lowess
# 1 Read and NAormalise the data
# 2 Generate W for the every query point
# 3 No training is involved directly make the predctions
# 4 where X is XT
# 5 find out the best value of tau bandwidth parameter


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dfx=pd.read_csv('')
dfy=pd.read_csv('')


# In[ ]:


X=dfx.values
Y=dfy.values


# In[ ]:


print(X.shape)
print(Y.shape)


# In[ ]:


# Normalize the data
u=X.mean()
std=X.std()
X=(X-u)/std


# In[ ]:


plt.title('Normalized data')
plt.scatter(X,Y)
plt.show()


# In[6]:


# find out W mXm diagonal Matrix
print(type(np.mat(np.eye(5))))


# In[ ]:


a*b ==> if a and b are matrix --> matrix multiplication
a*b --> if a and b are arrays --> Element wise


# In[ ]:


def getW(query_point,X,tau):
    M=X.shape[0]
    W=np.mat(np.eye(M))
    # W is the weight foe the training point
    for i in range(M):
        xi=X[i]
        x=query_point
        W[i,i]= np.exp(np.dot((xi-x),(xi-x).T)/(-2*tau*tau))
    return W


# In[ ]:


X=np.mat(X)
Y=np.mat(Y)
M=X.shape[0]

W=gotWeightMatrix(-1,X,1)
print(W.shape)
print(W)


# In[ ]:


if the tau square is very larget then it will approach to the identity matrix 


# # Making Predictions

# In[ ]:


def Predict(X,Y,query_x,tau):
    ones=np.ones((M,1))
    X_=np.hstack((X,ones))
    
    print(X_.shape)
    print(X_[:5,:])
    
    qx=np.mat([query_x,1])
    W=getW(qx,X_,tau)
    print(W.shape)
    
    theta=np.linalg.pinv(X_.T*(W*X))*(X_.T*(W*Y))
    print(theta.shape)
    pred=np.dot(qx,theta)
    return theta


# In[ ]:


theta,pred=predict(X,Y,1.0,1.0)
print(theta)
print(pred)

