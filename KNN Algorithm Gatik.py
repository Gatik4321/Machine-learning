#!/usr/bin/env python
# coding: utf-8

# In[8]:


# now we are going to implement the KNN algorithm
# simply wriritng the code for the knn algoerithm

import math


def classifyAPoint(points,p,k=3):
    distance=[]
    
    for group in points:
        for feature in points[group]:
            euclidiean_distance=math.sqrt((feature[0]-p[0])**2+(feature[1]-p[1])**2)
            distance.append((euclidiean_distance,group))
    distance=sorted(distance)[:k]
    
    freq0=0 # it is defined as the frequency of the group 0
    freq1=0 # it is defined as the frequency of the group 1
    
    for d in distance:
        if d[1]==0:
            freq0+=1
        elif d[1]==1:
            freq1+=1
            
    return 0 if(freq0>freq1) else 1
            
def main():
    #creating the dictionary of the points
    points={0:[(1,12),(2,5),(3,6),(3,10),(3.5,8),(2,11),(2,9),(1,7)],
           1:[(5,3),(3,2),(1.5,9),(7,2),(6,1),(3.8,1),(5.6,4),(4,2),(2,5)]}
    p=(2.5,7)
    k=3
    print("The value classified to unknown point is: {}".\
          format(classifyAPoint(points,p,k)))
if __name__=='__main__':
    main()


# In[9]:


# implementing the knn algorithm purely coded in python


# # KNN( K Nearest Neighbours)

# In[79]:


# Now implementin the KNN using the sklearn library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[80]:


# Loading the datatset to be implement ed on the knn
iris=datasets.load_iris()


# In[81]:


X=iris.data
Y=iris.target


# In[82]:


X


# In[83]:


Y


# In[84]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=100)


# In[85]:


X_train.shape


# In[86]:


X_test.shape


# In[87]:


Y_train.shape


# In[88]:


Y_test.shape


# In[89]:


# Now implementing the knnmodel o te iris dat
from sklearn.neighbors import KNeighborsClassifier


# In[90]:


knn3=KNeighborsClassifier(n_neighbors=3)
knn5=KNeighborsClassifier(n_neighbors=5)
knn7=KNeighborsClassifier(n_neighbors=7)


# In[91]:


# Prediction using KNN classifier
knn7.fit(X_train,Y_train)
knn5.fit(X_train,Y_train)
knn3.fit(X_train,Y_train)


# In[92]:


y_pred_7=knn7.predict(X_test)
y_pred_5=knn5.predict(X_test)
y_pred_3=knn3.predict(X_test)


# In[93]:


from sklearn.metrics import accuracy_score
print("Accuracy score with k=7",accuracy_score(Y_test,y_pred_7)*100)


# In[94]:


print("Accuracy score with k=5",accuracy_score(Y_test,y_pred_5)*100)


# In[95]:


print("Accuracy score with k=3",accuracy_score(Y_test,y_pred_3)*100)


# In[96]:


from sklearn.metrics import f1_score
f1_score(Y_test,y_pred_7,average='macro')


# In[97]:


from sklearn.metrics import f1_score
f1_score(Y_test,y_pred_5,average='macro')


# In[98]:


# Implementing the KNN algoeithm on the minsit dataset
from sklearn.metrics import f1_score
f1_score(Y_test,y_pred_3,average='macro')


# In[101]:


from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,y_pred_7,normalize='all')


# In[47]:


# importing the libraraies required for the minsit dataste
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


df=pd.read_csv('mnist_train.csv.csv')
print(df.shape)


# In[50]:


df.head()


# In[52]:


data=df.values


# In[54]:


print(data.shape)


# In[55]:


print(type(data))


# In[63]:


X=data[:,1:]
Y=data[:,0]


# In[64]:


print(X.shape)


# In[65]:


print(Y.shape)


# In[67]:


from sklearn.model_selection import train_test_split


# In[69]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,random_state=100)


# In[73]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[75]:


#visulalize some samples
def drawImg(sample):
    img=sample.reshape((28,28))
    plt.imshow(img,cmap='gray')
    #plt.show()
drawImg(X_train[190])


# In[76]:


# Can we apply knn on this data
# can we apply the KNN to this data
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

# Now writing the logic for the kNN
def knn(X,Y,queryPoint,k=5):
    vals=[]
    m=X.shape[0]
    
    for i in range(m):
        d=dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    vals=sorted(vals)
    vals=vals[:k]
    vals=np.array(vals)
    
    new_vals=np.unique(vals[:,1],return_counts=True)
    
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    
    return pred


# In[77]:


pred=knn(X_train,Y_train,X_test[0])
print(int(pred))


# In[78]:


drawImg(X_test[0])
print(Y_test[0])


# In[ ]:


from sklearn.metrics import accuracy_score

