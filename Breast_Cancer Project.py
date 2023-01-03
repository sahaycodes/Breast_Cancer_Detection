#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


#data collection & processing


# In[4]:


#loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()


# In[5]:


print(breast_cancer_dataset)


# In[6]:


#loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data ,columns = breast_cancer_dataset.feature_names)


# In[7]:


#print the first 5 rows of the dataframe 
data_frame.head()


# In[8]:


#adding the target column to the dataframe 
data_frame['label'] = breast_cancer_dataset.target


# In[9]:


#printing the last 5 rows of the dataframe
data_frame.tail()


# In[10]:


#no.of rows and columns in the dataset
data_frame.shape


# In[11]:


#getting some information about the data
data_frame.info()


# In[12]:


#checking for missing values
data_frame.isnull().sum()


# In[13]:


# statiscal measures about the data
data_frame.describe()


# In[14]:


#checking the distribution of target variables 
data_frame['label'].value_counts()


# In[15]:


#1 represents benign
#0 represents malign


# In[16]:


data_frame.groupby('label').mean()


# In[18]:


# separating the features and target
x = data_frame.drop(columns='label',axis=1)
y = data_frame['label']


# In[19]:


print(x)


# In[20]:


print(y)


# In[21]:


# splitting the data into training data and testing data 
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# In[22]:


print(x.shape,x_train.shape,x_test.shape)


# In[23]:


#model training 
#logistic regression model
model=LogisticRegression()


# In[24]:


#training the lr model using training data
model.fit(x_train,y_train)


# In[25]:


# model evaluation
#accuracy score 
#accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)


# In[26]:


print('ACCURACY ON TRAINING DATA =',training_data_accuracy)


# In[27]:


#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)


# In[28]:


print("accuracy on test data",test_data_accuracy)


# In[32]:


#building a predictive system
input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)


#change input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the numpy array as we are predicting for one datappoint 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('Breast Cancer Report = Malign')
else:
    print('Breast Cancer Report = Benign')


# In[ ]:


# end of Logistic Regression Model 
# end of Breast_Cancer Prediction 

