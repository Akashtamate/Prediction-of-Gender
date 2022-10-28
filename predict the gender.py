#!/usr/bin/env python
# coding: utf-8

# # **A classification model to predict the gender (male or female) based on different acoustic parameters**

# **IMPORTING ALL THE NECESSERY LIBRARIES**

# In[75]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# **DATA PREPROCESSING**

# In[2]:


df = pd.read_csv('/content/drive/MyDrive/voice.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().any()


# In[5]:


df.info()


# **Percentage distribution of labels on PIE CHART**

# In[8]:


new_df = df['label'].value_counts().rename_axis('Category').reset_index(name = 'Count')
new_df


# In[9]:


chart_labels = new_df.Category
chart_values = new_df.Count


# In[11]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
ax.pie(chart_values, labels = chart_labels, autopct = '%1.2f%%')
plt.show()


# In[12]:


#Assigning label 1 for male and label 0 for female
df.label = [1 if each == 'male' else 0 for each in df.label]
df.label.head()


# **Training and testing on various classifier models**
# 
# 

# In[17]:


x = df.drop(['label'], axis = 1)
y = df.label.values


# In[18]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[42]:


Algo_names = []
Algo_Accuracy = []


# **DECISION TREE CLASSIFIER**

# In[43]:


d_Tree = DecisionTreeClassifier(random_state = 42)
d_Tree.fit(x_train, y_train)
print("Accuracy of Decision Tree Classifier is: ", (d_Tree.score(x_test, y_test))*100)
Algo_names.append("Decision Tree Classifier")
Algo_Accuracy.append((d_Tree.score(x_test, y_test))*100)
pred1 = d_Tree.predict(x_test)


# **Confusion matrix for Decision tree classifier Model**

# In[77]:


DTC = confusion_matrix(y_test, pred1)
plt.figure(figsize = (5, 5))
sns.heatmap(DTC, annot = True, fmt = ".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix for Decision Tree Classifier")
plt.show()


# **Classification report of Decision Tree Classifier Model**

# In[68]:


report_d_Tree = classification_report(y_test, pred1)
print(report_d_Tree)


# **RANDOM FOREST CLASSIFER**

# In[44]:


r_Forest = RandomForestClassifier(random_state = 42)
r_Forest.fit(x_train, y_train)
print("Accuracy of Random Forest Classifer is: ", (r_Forest.score(x_test, y_test))*100)
Algo_names.append("Random Forest Classifer")
Algo_Accuracy.append((r_Forest.score(x_test, y_test))*100)
pred2 = r_Forest.predict(x_test)


# **Confusion matrix for Random Forest Classifier Model**

# In[79]:


RFC = confusion_matrix(y_test, pred2)
plt.figure(figsize = (5, 5))
sns.heatmap(RFC, annot = True, fmt = ".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix for Random Forest Classifier")
plt.show()


# **Classification report of Random Forest Classifier Model**

# In[69]:


report_r_Forest= classification_report(y_test, pred2)
print(report_r_Forest)


# **KNN CLASSIFER**

# In[45]:


KNN = KNeighborsClassifier(n_neighbors = 5)
KNN.fit(x_train, y_train)
print("Accuracy of KNN Classifer is: ", (KNN.score(x_test, y_test))*100)
Algo_names.append("KNN Classifer")
Algo_Accuracy.append((KNN.score(x_test, y_test))*100)
pred3 = KNN.predict(x_test)


# **Confusion matrix for KNN Classifier Model**

# In[80]:


KNN_confusion = confusion_matrix(y_test, pred3)
plt.figure(figsize = (5, 5))
sns.heatmap(KNN_confusion, annot = True, fmt = ".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix for KNN Classifier")
plt.show()


# **Classification report of KNN Classifier Model**

# In[70]:


report_KNN= classification_report(y_test, pred3)
print(report_KNN)


# **LOGISTIC REGRESSION**

# In[46]:


LR = LogisticRegression(random_state = 42, max_iter = 1000)
LR.fit(x_train, y_train)
print("Accuracy of Logistic Regression is: ", (LR.score(x_test, y_test))*100)
Algo_names.append("Logistic Regression")
Algo_Accuracy.append((LR.score(x_test, y_test))*100)
pred4 = LR.predict(x_test)


# **Confusion matrix for Logistic Regression Model**

# In[81]:


LR_confusion = confusion_matrix(y_test, pred4)
plt.figure(figsize = (5, 5))
sns.heatmap(LR_confusion, annot = True, fmt = ".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix for Logistic Regression")
plt.show()


# **Classification report of Logistic Regression Model**

# In[71]:


report_LR= classification_report(y_test, pred4)
print(report_LR)


# **SVM CLASSIFER**

# In[47]:


svm = SVC(random_state = 42)
svm.fit(x_train, y_train)
print("Accuracy of SVM Classifer is: ", (svm.score(x_test, y_test))*100)
Algo_names.append("SVM classifer")
Algo_Accuracy.append((svm.score(x_test, y_test))*100)
pred5 = svm.predict(x_test)


# **Confusion matrix for SVM Classifier Model**

# In[83]:


SVM_confusion = confusion_matrix(y_test, pred5)
plt.figure(figsize = (5, 5))
sns.heatmap(SVM_confusion, annot = True, fmt = ".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion matrix for SVM Classifier")
plt.show()


# **Classification report of SVM model**

# In[72]:


report_svm= classification_report(y_test, pred5)
print(report_svm)


# **Comparisions of accuracy(scores) of all the classifier models**

# In[66]:


plt.figure(figsize = (11, 5))
plt.ylim(55, 100)
plt.bar(Algo_names, Algo_Accuracy, width = 0.2, color = ['magenta'])
plt.xlabel("Algorithm names")
plt.ylabel("Algorithm accuracy")
plt.show() 


# # **CONCLUSION**
# 
# ## From the above Bar Chart, which shows the accuracy of various classifier models, It is pretty evidient that ***Random Forest Classifier performs best with an accuracy of 98%*** for the given dataset in comparision to other classifier models.
