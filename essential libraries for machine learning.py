#!/usr/bin/env python
# coding: utf-8

# In[20]:

#[Author] : Muhammed Eldabea Hashem tarks 

import numpy as np 


# In[2]:


my_array=np.array([1,2,3,4,5,5,6])


# In[4]:


print(my_array.shape
     )


# In[3]:


print(np.zeros((3,3)))


# In[4]:


print(np.ones((3,3)))


# In[5]:


print(np.full((3,3),7))


# In[8]:


print(np.eye(3)) #creat identity matrix


# In[11]:


print(np.random.random((3,3)))


# In[18]:


x=np.array([[1,2,3],[4,5,6]])
y=np.array([[12,13,14],[15,16,17]])
z=np.array([[12,13,14],[15,16,17],[1,2,3]])


# In[16]:


print(x+y)
print()
print(np.add(x,y))


# In[19]:


print(x.dot(z))


# In[21]:


x_transpose=x.transpose()
print(x_transpose)


# In[22]:


print(x_transpose)


# In[23]:


import matplotlib.pyplot as plt 


# In[34]:


f=np.arange(0,3*np.pi,0.1)
a=np.sin(f)
v=np.cos(f)


# In[41]:


plt.subplot(2,1,1)
plt.plot(f,a)
plt.title("sine" )
#plt.legend(["sine","cosine"])
plt.subplot(2,1,2)
plt.plot(f,v)
plt.title("cosine" )

plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.show()


# In[34]:


import imageio as im
import skimage as skim


# In[28]:


img=im.imread(r"C:\Users\tahat\Pictures\4fd58e90ffd8cbafb72fba60ae9a4f26-700.jpg")
print(img.dtype,img.shape)


# In[29]:


#(size,chanell(RGB/BW))


# In[37]:


img=im.imread(r"C:\Users\tahat\Pictures\4fd58e90ffd8cbafb72fba60ae9a4f26-700.jpg")
#imga= skim.transform.resize(img,(500,500))
imgcoler=img*[0,0.9,0.8]


# In[38]:


im.imsave(r"C:\Users\tahat\Pictures\maio.jpg",imgcoler)


# In[39]:


import tensorflow as tf 


# In[43]:


a=2 
b=4
c=tf.add(a,b,name='add')
#we need to gice our operation a name as we eal with a session her not a normal operation 


# In[42]:


print(c)


# In[ ]:




