#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf

import matplotlib.pyplot as plt


# In[3]:


mnist = tf.keras.datasets.mnist


# In[4]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[5]:


plt.imshow(x_train[4],cmap=plt.cm.binary)
plt.show()


# In[6]:


x_train[0]


# In[ ]:





# In[7]:


x_train= tf.keras.utils.normalize(x_train,axis=1)
x_test= tf.keras.utils.normalize(x_test,axis=1)


# In[8]:


x_train[0]


# In[ ]:





# In[9]:


model=tf.keras.models.Sequential() #a feed forward model
model.add(tf.keras.layers.Flatten()) #takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #a simple fully connected layer
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) # our output layer. 10 units for 10 classes. Softmax for probability distribution


# In[10]:


model.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy', #how will we calculate the error to minimize the loss
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10)


# In[11]:


val_loss,val_acc=model.evaluate(x_test,y_test)


# In[12]:


val_loss


# In[13]:


val_acc


# In[14]:


model.save(r'C:\Python37\Projects\Digit Recognition\digit_model.model')

#Training Finished!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# From here Testing Starts---------------------------->

new_model=tf.keras.models.load_model(r'C:\Python37\Projects\Digit Recognition\digit_model.model')
predictions=new_model.predict(x_test)


# In[16]:


predictions[0]


# In[17]:


import numpy as np


# In[18]:


plt.imshow(x_test[30],cmap=plt.cm.binary)
plt.show()


# In[19]:


np.argmax(predictions[30])


# In[ ]:




