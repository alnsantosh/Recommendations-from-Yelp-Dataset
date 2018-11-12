
# coding: utf-8

# In[18]:


# Load the library with the iris dataset
from sklearn.datasets import load_iris

# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)


# In[37]:


from collections import defaultdict
import math


# In[57]:


reader=pd.read_csv('F://topical.txt',delimiter="\t",header=None,names=["business_id","business_stars","business_review_count","review_id","review_stars","text","user_id","name","user_review_count","yelping_since","friends","useful","funny","cool","fans","elite"],encoding='latin-1')


# In[58]:


#reader.head()


# In[59]:


f=defaultdict(dict)
for i in range(len(reader)):
    f[reader["user_id"][i]]={};
    


# In[60]:


print(f[reader["user_id"][2]])
print(len(f))
print(len(reader))
#reader=reader[:5]


# In[ ]:


for i in range(len(reader)):
    if(i%10000==0):
        print(i)
    if "review_count" in f[reader["user_id"][i]]:
        f[reader["user_id"][i]]["review_count"]+=1
    else:
        f[reader["user_id"][i]]["review_count"]=1
        
    if "ratings" in f[reader["user_id"][i]]:
        f[reader["user_id"][i]]["ratings"].append(reader["review_stars"][i])
    else:
        f[reader["user_id"][i]]["ratings"]=[]
        f[reader["user_id"][i]]["ratings"].append(reader["review_stars"][i])
        
    if "business_reviewed" in f[reader["user_id"][i]]:
        f[reader["user_id"][i]]["business_reviewed"][reader["business_id"][i]]=1
    else:
        #print("business_reviewed")
        f[reader["user_id"][i]]["business_reviewed"]={}
        f[reader["user_id"][i]]["business_reviewed"][reader["business_id"][i]]=1
        
    if "yelping_age" not in f[reader["user_id"][i]]:
#         if(type(reader["yelping_since"][i])==float and math.isnan(reader["yelping_since"][i])):
#             f[reader["user_id"][i]]["yelping_age"]=0
#         else:
#             print("Failed: ",i)
#             f[reader["user_id"][i]]["yelping_age"]= 2018-int(reader["yelping_since"][i][0:4])
        try:
            f[reader["user_id"][i]]["yelping_age"]= 2018-int(reader["yelping_since"][i][0:4])
        except:
            f[reader["user_id"][i]]["yelping_age"]=0
    #print(reader["user_id"][i],": ",f[reader["user_id"][i]])
    
        


# In[66]:


print(type(reader["yelping_since"][39721]))


# In[12]:


reader['is_train']=np.random.uniform(0,1,len(reader)) <=0.75
#reader.head()


# In[7]:


train=reader[reader['is_train']==True]
test=reader[reader['is_train']==False]
print(len(train))
print(len(test))
print(len(train)/(len(train)+len(test)))


# In[8]:


features=reader.columns[:]
features


# In[10]:


y=[]
for i in range(len(reader)):
    if(reader["elite"] is None):
        y.append(0)
    else:
        y.append(0);


# In[14]:


print(y)


# In[23]:


clf=RandomForestClassifier(n_jobs=2,random_state=0)
#clf.fit(reader["business_review_count"],y)


# In[ ]:


#clf.predict(test[features])

