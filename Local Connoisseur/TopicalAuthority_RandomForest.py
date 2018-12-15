
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


# In[3]:


feature_df=pd.read_csv("F://topical_features.txt")


# In[4]:


print(len(feature_df))
feature_df.head()


# In[5]:


feature_df_new=feature_df[:]
print(len(feature_df_new[feature_df_new['y']==1]))
print(len(feature_df_new[feature_df_new['y']==0]))
feature_1=feature_df_new[feature_df_new['y']==1]
feature_0=feature_df_new[feature_df_new['y']==0][:50000]
print(len(feature_1))
print(len(feature_0))


# In[6]:


feature_nonskew=feature_1.append(feature_0)
print(len(feature_0))
print(len(feature_1))
print(len(feature_nonskew))
feature_nonskew.head()


# In[7]:


#RandomForest on Non-Skewed data
feature_nonskew['is_train']=np.random.uniform(0,1,len(feature_nonskew)) <=0.75
train=feature_nonskew[feature_nonskew['is_train']==True]
test=feature_nonskew[feature_nonskew['is_train']==False]
print(len(train))
print(len(test))
features=feature_nonskew.columns[1:-2]
y=train["y"]


# In[8]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None)
clf.fit(train[features],y)


# In[9]:


test_y=clf.predict(test[features])


# In[10]:


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


# In[11]:


clf=RandomForestClassifier(random_state=0,n_estimators=100,max_depth=None)
clf.fit(train[features],y)


# In[12]:


test_y=clf.predict(test[features])


# In[13]:


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


# In[35]:


clf=RandomForestClassifier(random_state=0,n_estimators=50,max_depth=None)
clf.fit(train[features],y)


# In[36]:


test_y=clf.predict(test[features])


# In[37]:


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


clf=RandomForestClassifier(random_state=0,n_estimators=1,max_depth=None)
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


# In[17]:


clf=RandomForestClassifier(random_state=0,n_estimators=200,max_depth=None)
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


# In[53]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None,max_leaf_nodes=50)
clf.fit(train[features],y)


# In[54]:


test_y=clf.predict(test[features])


# In[55]:


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


# In[56]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None,max_leaf_nodes=75)
clf.fit(train[features],y)


# In[57]:


test_y=clf.predict(test[features])


# In[58]:


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


# In[59]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None,max_leaf_nodes=25)
clf.fit(train[features],y)


# In[60]:


test_y=clf.predict(test[features])


# In[61]:


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


# In[62]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None,max_leaf_nodes=100)
clf.fit(train[features],y)


# In[63]:


test_y=clf.predict(test[features])


# In[64]:


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

