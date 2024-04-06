#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mglearn


# In[3]:


import mglearn


# In[4]:


mglearn.plots.plot_knn_classification(n_neighbors=1)


# In[5]:


mglearn.plots.plot_knn_classification(n_neighbors=3)


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


x,y=mglearn.datasets.make_forge()
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
snf=KNeighborsClassifier(n_neighbors=3)
snf.fit(x_train,y_train)


# In[8]:


snf.predict(x_test)


# In[9]:


snf.score(x_test,y_test)


# In[10]:


from sklearn.datasets import load_breast_cancer


# In[11]:


kanser=load_breast_cancer()


# In[12]:


kanser.keys()


# In[13]:


print (kanser['DESCR'])


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(kanser.data, kanser.target, stratify=kanser.target, random_state=66)


# In[15]:


egitim_dogruluk=[]
test_dogruluk=[]


# In[16]:


komsuluk_sayisi=range(1,11)


# In[17]:


import matplotlib.pyplot as plt

for n_komsuluk in komsuluk_sayisi:
    snf=KNeighborsClassifier(n_neighbors=n_komsuluk)
    snf.fit(x_train,y_train)
    egitim_dogruluk.append(snf.score(x_train,y_train))
    test_dogruluk.append(snf.score(x_test,y_test))
    
plt.plot(komsuluk_sayisi, egitim_dogruluk, label='Eğitim doğruluk')
plt.plot(komsuluk_sayisi, test_dogruluk, label='Test doğruluk')
plt.ylabel('Doğruluk')
plt.xlabel('n-komşuluk')
plt.legend()


# In[18]:


mglearn.plots.plot_knn_regression(n_neighbors=1)


# In[19]:


mglearn.plots.plot_knn_regression(n_neighbors=3)


# In[21]:


from sklearn.neighbors import KNeighborsRegressor


# In[22]:


x,y=mglearn.datasets.make_wave(n_samples=40)


# In[23]:


x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0)


# In[24]:


reg=KNeighborsRegressor(n_neighbors=3)


# In[25]:


reg.fit(x_train,y_train)


# In[26]:


reg.score(x_test,y_test)


# In[ ]:




