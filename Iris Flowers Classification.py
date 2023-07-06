#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Load the Iris dataset
iris = load_iris()


# In[7]:


# Create a DataFrame to store the dataset
df = pd.DataFrame(iris.data, columns=iri.feature_names)
df['species'] = iris.target_names[iris.target]


# In[8]:


df.head()


# In[9]:


# Split the data into features (X) and target variable (y)
X = df.drop('species', axis=1)
y = df['species']


# In[10]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model = LogisticRegression(max_iter=1000)


# In[16]:


# Train the model
model.fit(X_train, y_train)


# In[17]:


# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# In[18]:


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)


# In[ ]:




