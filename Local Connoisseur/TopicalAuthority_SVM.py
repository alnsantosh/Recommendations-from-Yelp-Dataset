
# coding: utf-8

# In[1]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)

from collections import defaultdict
import math
import pickle


# In[2]:


feature_df=pd.read_csv("F://topical_features.txt")


# In[3]:


print(len(feature_df))
feature_df.head()


# In[11]:


feature_df_new=feature_df[:]
print(len(feature_df_new[feature_df_new['y']==1]))
print(len(feature_df_new[feature_df_new['y']==0]))
feature_1=feature_df_new[feature_df_new['y']==1]
feature_0=feature_df_new[feature_df_new['y']==0][:50000]
print(len(feature_1))
print(len(feature_0))


# In[12]:


feature_nonskew=feature_1.append(feature_0)
print(len(feature_0))
print(len(feature_1))
print(len(feature_nonskew))
feature_nonskew.head()


# In[13]:


#RandomForest on Non-Skewed data
feature_nonskew['is_train']=np.random.uniform(0,1,len(feature_nonskew)) <=0.75
train=feature_nonskew[feature_nonskew['is_train']==True]
test=feature_nonskew[feature_nonskew['is_train']==False]
print(len(train))
print(len(test))
features=feature_nonskew.columns[1:-2]
y=train["y"]


# In[14]:


clf=svm.SVC(gamma=0.001)
clf.fit(train[features],y)


# In[15]:


test_y=clf.predict(test[features])


# In[16]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[17]:


clf=svm.SVC()
clf.fit(train[features],y)


# In[18]:


test_y=clf.predict(test[features])


# In[19]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[20]:


clf=svm.SVC(C=0.1)
clf.fit(train[features],y)


# In[21]:


test_y=clf.predict(test[features])


# In[22]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[23]:


clf=svm.SVC(C=0.1,tol=0.005)
clf.fit(train[features],y)


# In[25]:


test_y=clf.predict(test[features])


# In[26]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[27]:


clf=svm.SVC(gamma=0.25,C=0.1,tol=0.005)
clf.fit(train[features],y)


# In[28]:


test_y=clf.predict(test[features])


# In[29]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[30]:


clf=svm.SVC(gamma=0.4,C=0.1,tol=0.005)
clf.fit(train[features],y)


# In[31]:


test_y=clf.predict(test[features])


# In[32]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)


# In[33]:


clf=svm.SVC(gamma=0.5,C=0.1,tol=0.005)
clf.fit(train[features],y)


# In[34]:


test_y=clf.predict(test[features])


# In[35]:


count=0
count_1=0
count_actual_1=0
y_given=list(test["y"])
print(type(y_given))
for i in range(len(test)):
    if(y_given[i]==test_y[i]):
        count+=1
    if(y_given[i]==test_y[i] and test_y[i]==1):
        count_1+=1
    if(y_given[i]==1):
        count_actual_1+=1
print(count/len(test))
print(count_1/(count_actual_1))
print(count_1)
print(count_actual_1)
print(count_1)
fp=0
fn=0
for i in range(len(y_given)):
    if(y_given[i]==0 and test_y[i]==1):
        fp+=1
    if(y_given[i]==1 and test_y[i]==0):
        fn+=1
print(fp)
tp=count_1
precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print("Precision: ",precision)
print("Recall: ",recall)
print("F-Measure: ",f_measure)

