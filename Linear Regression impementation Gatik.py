#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the required libraries for the linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Implementatio of the gradient descent in general
X=np.arange(10)
Y=(X-5)**2
print(X,Y)


# In[5]:


## Given a function f(x) we have to find the value of x that minimizes it
## finding the function that minimizes the f
plt.style.use('seaborn')
plt.plot(X,Y)
plt.ylabel('F(x)')
plt.xlabel('X')
plt.show()


# In[6]:


# Now we know that find the minimum value for the x by using the gradient descent algorithm
x=0
lr=0.1
error=[] # we want to store the error of the each iteration in it
plt.plot(X,Y)

for i in range(50):
    grad=2*(x-5)
    x=x-lr*grad
    y=(x-5)**2
    error.append(y)
    plt.scatter(x,y)
    print(x)


# In[7]:


# Now seeing how the value for the error change overtime
plt.plot(error)
plt.show()


# # Now we going to apply the Linear Regression on the kaggle dataset

# In[74]:


# data preparation challenege for the hardwork pays off
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[73]:


# Now loading the csv files for the dataset
X=pd.read_csv('Linear_X_Train.csv')
Y=pd.read_csv('Linear_Y_Train.csv')


# In[19]:


X


# In[20]:


Y


# In[21]:


plt.style.use('seaborn')
plt.scatter(X,Y)
plt.title("Time vs Performance graph")
plt.xlabel("Hardwork")
plt.ylabel("Performance")
plt.show()


# In[22]:


X.shape


# In[23]:


Y.shape


# In[24]:


# convert the X and Y into the numpy arrays
X=X.values


# In[29]:


Y=Y.values


# In[28]:


X


# In[30]:


Y


# In[33]:


# Now we will going to normaliza the data
u=X.mean()
std=X.std()
X=X-u/std


# In[34]:


X


# In[35]:


u


# In[36]:


std


# In[37]:


plt.style.use('seaborn')
plt.scatter(X,Y)
plt.title("Time vs Performance graph")
plt.xlabel("Hardwork")
plt.ylabel("Performance")
plt.show()


# In[53]:


# Now writing the complete linear Regression function purely in python
def hypothesis(x,theta):
    y_=theta[0]+theta[1]*x
    return y_
# Now we are writin the function to fins the gradient
def gradient(X,Y,theta):
    m=X.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        y_=hypothesis(X[i],theta)
        y=Y[i]
        grad[0]+=(y_-y)
        grad[1]+=(y_-y)*X[i]
    return grad/m
# function that will compute the total error
def error(X,Y,theta):
    m=X.shape[0]
    total_error=0.0
    for i in range(m):
        y_=hypothesis(X[i],theta)
        total_error+=(y_-Y[i])**2
    return total_error/m
# Now we will going to find the gradient descent
def gradientDescent(X,Y,max_steps=100,learning_rate=0.1):
    theta=np.zeros((2,))
    error_list=[]
    theta_list=[]
    for i in range(max_steps):
        e=error(X,Y,theta)
        error_list.append(e)
        # compute the gradient
        grad=gradient(X,Y,theta)
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
    return theta,error_list,theta_list


# In[54]:


theta,error_list,theta_list=gradientDescent(X,Y)


# In[58]:


theta


# In[57]:


error_list


# In[60]:


# Now plotting the error list
plt.plot(error_list)
plt.title("Reduction error overtime")
plt.show()


# # Section - 3 Predictions and Best Line

# In[62]:


y_=hypothesis(X,theta)
print(y_)


# In[63]:


# Training and Predictions
plt.scatter(X,y)
plt.plot(X,y_,color='orange',label="Predictions0")
plt.legend()
plt.show()


# In[64]:


# Loading the test data
X_test=pd.read_csv("11.csv")
Y_test=pd.read_csv("14.csv")


# In[65]:


X_test.shape


# In[66]:


Y_test.shape


# In[67]:


y_test=hypothesis(X_test,theta)


# In[69]:


plt.plot(X_test,Y_test)
plt.scatter(X_test,y_test)


# In[75]:


df=pd.Dataframe(data=y_test,columns=["y"])


# # Section -4 Computing the score

# In[76]:


def r2_score(Y,y_):
    # instead of using loop np.sum() is recommended
    num=np.sum((Y-y_)**2)
    denom=np.sum((Y-Y.mean())**2)
    score=1-num/denom
    return score*100


# In[77]:


r2_score(Y,y_)


# In[78]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# In[79]:


theta


# # Section -5 Visualising Loss function, Gradient Descent Trajectory, theta Updates

# In[81]:


# Surface Plot are used to visuallise Loss function in Machine Learning and Deep Learning
a = np.arange(-1,1,0.02)
b = a 
a,b = np.meshgrid(a,b)


# In[115]:


fig = plt.figure()
axes = fig.gca(projection='3d')
axes.plot_surface(a,b,a**2+b**2,cmap='rainbow')
plt.show()


# In[116]:


# Now we are going to implement the linear regression using sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[117]:


X=pd.read_csv("Linear_X_Train.csv")
Y=pd.read_csv("Linear_Y_Train.csv")


# In[118]:


X


# In[119]:


Y


# In[120]:


from sklearn.linear_model import LinearRegression


# In[121]:


lr=LinearRegression()


# In[122]:


X.shape


# In[123]:


Y.shape


# In[124]:


from sklearn.model_selection import train_test_split


# In[125]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=101)


# In[126]:


X_train.shape


# In[127]:


X_test.shape


# In[128]:


Y_train.shape


# In[129]:


Y_test.shape


# In[130]:


lr.fit(X_train,Y_train)


# In[133]:


y_pred=lr.predict(X_test)


# In[134]:


print(y_pred)


# In[135]:


c = [i for i in range (1,len(Y_test)+1,1)]
plt.plot(c,Y_test,color='r',linestyle='-')
plt.plot(c,y_pred,color='b',linestyle='-')
plt.xlabel('A')
plt.ylabel('B')
plt.title('Prediction vs actual')
plt.show()


# In[136]:


# Now calculating the r2 score for evaluating the how good is our model


# In[137]:


from sklearn.metrics import r2_score 
r2_score(Y_test,y_pred)

