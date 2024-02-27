#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install seaborn')


# In[28]:


import seaborn as sns
import numpy as np


# In[61]:


# seaborn also has its own inbuilt data set  ex tips titanic
tips=sns.load_dataset('tips')


# In[62]:


tips


# In[31]:


tips.head(10)


# In[33]:


# drawing bar grpah using the seaborn librar
sns.barplot(x='sex',y='total_bill',data= tips,estimator=np.std)


# In[16]:


# understanding the countplot in seaborn
# we cannot change the agrgation function in the count plot
sns.countplot(x='sex',data=tips)


# In[17]:


sns.countplot(x='time',data=tips)


# In[18]:


tips.dtypes


# In[34]:


sns.boxplot(x="day",y="total_bill",data=tips,hue="smoker")
# box plot are basically used to find the ouliers


# In[35]:


sns.violinplot(x='day',y='total_bill',data=tips,hue='sex',split=True)


# In[47]:


# Now we going to study the distribution plot
# it is basically used to study about the spread of one variable
# it tell us the info about only one variable
sns.histplot(tips['total_bill'],kde=True,bins=40)
# the line is called kde we can remove the line simply by kde=false


# In[44]:


# if you just want to plot the line then the graph is called kdeplot
sns.kdeplot(tips['total_bill'])


# In[53]:


# join plot is used to see the relationship bewteen twovariables
sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')#scatter ,kde


# In[55]:


# if we want to show the relation betweene all the variables then we use pairplot
sns.pairplot(tips,hue="sex")


# In[56]:


# Now knowing about the more


# In[57]:


tips.head()


# In[64]:


print(tips.corr())
# if we want to show the correlation betweeen the variables then we can use this
tips_corr=tips.corr()
print(tips_corr)
# 1 means that variables are highly correlated with each other
# 0 means that variables are not corrleadted oppsie


# In[65]:


# heatmap can only be drawn if we have vriables on both side x and y
# thats why we leanr the concept of correlation
sns.heatmap(tips_corr,annot=True)
# if we pass annot=True we cn able to see numbers


# In[66]:


flights=sns.load_dataset('flights')


# In[67]:


flights


# In[69]:


sns.countplot(x="passengers",data=flights)


# In[71]:


# we learn how to create the pivot table in pandas basically
# it is used to change the data set
fll=flights.pivot_table(index="month",columns="year",values="passengers")
print(fll)
# Heatmap is basically used to show the correlation


# In[74]:


sns.heatmap(fll,cmap="coolwarm")


# In[ ]:




