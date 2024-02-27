#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np


# In[3]:


themes=plt.style.available
print(themes)


# In[26]:


plt.style.use('seaborn-v0_8-darkgrid')


# In[27]:


# line plot
x=np.arange(10)
y1=x**2
y2=2*x+3
print(x)
print(y1)
print(y2)


# In[40]:


plt.plot(x,y1,color='red',label="Apple",marker='o',markersize=8)
plt.plot(x,y2,color='green',label="Kiwi",linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Prices of fruits over Time")
# this is used to show that corner box
plt.legend()
# it will simply print the graph
plt.show()


# In[4]:


get_ipython().run_line_magic('pinfo', 'plt.plot')


# In[42]:


import matplotlib
print(matplotlib.__version__)


# In[45]:


themes=plt.style.available
print(themes)


# In[46]:


plt.style.use('grayscale')


# In[47]:


# this is how we can draw a pyplot using numpy
import matplotlib.pyplot as plt
import numpy as np
x=np.array([0,6])
y=np.array([0,250])
plt.plot(x,y)


# In[5]:


import matplotlib.pyplot as plt
import numpy as np
xpoints=np.array([1,8,12,13,14])
ypoints=np.array([3,10,11,12,13])
plt.plot(xpoints,ypoints)


# In[51]:


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10, 5, 7])

plt.plot(ypoints)
plt.show()


# In[6]:


get_ipython().run_line_magic('pinfo', 'plt.plot')



# In[7]:


#We can use the default marker keyword for plotting the graph
import matplotlib.pyplot as plt
import numpy as np
ypoints=np.array([3,8,1,10])
plt.plot(ypoints,marker='O')
plt.show()


# In[83]:


prices=np.array([1,2,3,4])**3
print(prices)
plt.plot(prices)
plt.show()


# # Now we Learn how to draw scatter plots

# In[85]:


# this is how we can draw scatter plots
import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([6,7,8,9,10])
# dotted plot woithout line is called scatter plot
plt.scatter(x,y)
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Example data for x, y1, and y2
x = [1, 2, 3, 4, 5]
y1 = [10, 12, 8, 15, 11]
y2 = [5, 8, 6, 10, 7]

# Scatter plot for Apple prices
plt.scatter(x, y1, color='red', label="Apple", marker='o', markersize=8)

# Scatter plot for Kiwi prices
plt.scatter(x, y2, color='green', label="Kiwi", marker='x', markersize=8)

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Prices of fruits over Time")

# Add linestyle parameter to modify the line style for Kiwi prices
# plt.plot(x, y2, color='green', linestyle="da?shed")

# Show legend
plt.legend()

# Display the plot
plt.show()


# In[9]:


# import matplotlib.pyplot as plt
import numpy as np
x=np.arange(10)
y1=x**2
y2=2*x+3
print(x)
print(y1)
print(y2)
# this is how we adjust the size of any plot
plt.figure(figsize=(6,6))
plt.scatter(x,y1,color='r',label='Apple',marker='o')
plt.scatter(x,y2,color='green',label='Kiwi',linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Scatter plots-Prices of fruits over time")
plt.legend()
plt.show()


# # Bar graph

# In[101]:


get_ipython().run_line_magic('pinfo', 'plt.Bar')


# In[102]:


get_ipython().run_line_magic('pinfo', 'plt.bar')


# In[124]:


plt.style.use('dark_background')


# In[10]:


x_coordinates=np.array([0,1,2])*2
plt.bar(x_coordinates-.25,[10,20,15],color='b',width=0.5,label="Current Year",tick_label=["Gold","Platinum","Silver"])# Current Year
plt.bar(x_coordinates+.25,[20,10,12],color='y',width=0.5,label="Next Year",)# Next Year
plt.title("Metal Price Comparison")
plt.xlabel("Metal")
plt.ylabel("Price")
plt.ylim(0,40)
plt.xlim(0,5)
plt.legend()
plt.show()


# In[117]:


get_ipython().run_line_magic('pinfo', 'plt.bar')


# In[123]:


themes=plt.style.available
print(themes)


# In[ ]:


plt.style.use('dark_background')


# # Now we learn how to draw pie chart using matplot lib

# In[131]:


get_ipython().run_line_magic('pinfo', 'plt.pie')


# In[140]:


subjects=["Maths","Chem","Physics","English"]
weightage=[20,10,15,5]
plt.pie(weightage,labels=subjects,explode=(0,0,.1,0),startangle=90,shadow=True,autopct='%1.1f%%')
plt.legend(title="Four Fruits:")
plt.show()
# the property with the name autopcg used to get percentages


# In[11]:


get_ipython().run_line_magic('pinfo', 'plt.pie')


# In[22]:


plt.style.use('dark_background')
subjects="Maths","Chem","Physics","English"
weightage=[20,10,15,5]     
mycolors = ["Red", "hotpink", "b", "#4CAF50"]
plt.pie(weightage,labels=subjects,explode=(0,0,.1,0),shadow=True,autopct='%1.1f%%',colors=mycolors)
plt.legend(title="four fruits:")
plt.show()


# # Now we going to study about the histogram

# In[28]:


# random.radn returns the value of the standard nomal distrubution


# In[34]:


xsn=np.random.randn(100)
sigma=9;
u=60
X1=np.round(xsn*sigma+u)
X2=np.round(xsn*5+40)
print(X)
# print(xsn)


# In[35]:


themes=plt.style.available
print(themes)


# In[41]:


# Normal distribution and standard normal distribution is linked to the histogram
# Histogram used to show no. of attributes present in that particular rnge
plt.style.use('seaborn-v0_8-deep')
plt.hist(X1,label="Physics")
plt.hist(X2,alpha=.8,label="Maths")
plt.xlabel("Marks Range")
plt.ylabel("Prob/FreqCount of students")
plt.title("Histogram")
plt.legend()
plt.show()
plt.show()


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
# understanding the concept of the normal distributio
# Generate a Normal Distribution (1-D)
u=5
sigma=1
# when u=0 and sigma=1 such distribution is called SND
# standard normal distribution

vals=u+sigma*np.random.randn(100)
print(vals.shape)

plt.hist(vals,100)
plt.show()


# In[15]:


vals=np.round(vals)
z=np.unique(vals,return_counts=True)
print(z)


# In[16]:


x=vals
y=np.zeros(x.shape)

plt.scatter(x,y)
plt.show()


# In[ ]:




