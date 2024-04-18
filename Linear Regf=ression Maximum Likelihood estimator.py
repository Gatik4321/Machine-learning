#!/usr/bin/env python
# coding: utf-8

# In[13]:


# importing the requird libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


# If our data is look like regression data this means our assumption is correct
X=np.arange(20)
print(X)


# In[15]:


Theta=np.array([2,3])


# In[20]:


noise=np.random.randn(20)
print(noise)
y_ideal=Theta[1]*X+Theta[0]
y_real=Theta[1]*X+Theta[0]+noise
y=noise


# In[22]:


plt.plot(X,y_ideal,color='orange')
plt.scatter(X,y_real)
plt.show()


# In[ ]:





# In[ ]:




