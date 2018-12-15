
# coding: utf-8

# In[2]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Set random seed
np.random.seed(0)


# In[3]:


from collections import defaultdict
import math
import pickle


# In[4]:


reader=pd.read_csv('F://topical.txt',delimiter="\t",header=None,names=["business_id","business_stars","business_review_count","review_id","review_stars","text","user_id","name","user_review_count","yelping_since","friends","useful","funny","cool","fans","elite"],encoding='latin-1')


# In[5]:


reader.head()


# In[6]:


f=defaultdict(dict)
for i in range(len(reader)):
    f[reader["user_id"][i]]={};
    


# In[7]:


print(f[reader["user_id"][2]])
print(len(f))
print(len(reader))
#reader=reader[:5]


# In[8]:


#creating features
for i in range(len(reader)):
    if(i%100000==0):
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
    
        


# In[9]:


avg_yelp_age=0;
count=0
for key in f:
    if("yelping_age" in f[key] and f[key]["yelping_age"] !=0):
        avg_yelp_age+=f[key]["yelping_age"]
        count+=1
avg_yelp_age=avg_yelp_age/count;

count=0
remove_keys=[]
for key in f:
    try:
        if(type(key)==float and math.isnan(key)):
            continue
#         print(f[key]["ratings"])
#         print(key)
        rate=np.array(list(map(int,f[key]["ratings"])))
    #     print(f[key]["ratings"])
    #     print(key)
        avg=np.mean(rate)
        std=np.std(rate)
        f[key]["avg_rating"]=avg
        f[key]["stdev"]=std
        f[key]["unique_business_count"]=len(f[key]["business_reviewed"])
        if(f[key]["yelping_age"] ==0):
            f[key]["yelping_age"] =avg_yelp_age
        count+=1
        if(count%50000==0):
            print(f[key]["avg_rating"]," ",f[key]["stdev"]," ",f[key]["unique_business_count"])
    except:
        remove_keys.append(key)
    


# In[10]:


# print(f["-3i9bhfvrM3F1wsC9XIB8g"])
# print(f["-3i9bhfvrM3F1wsC9XIB8g"]["ratings"])
# print(len(remove_keys))

for key in remove_keys:
    f.pop(key,None)


# In[11]:


# print(type(reader["yelping_since"][39721]))
print(f["-3i9bhfvrM3F1wsC9XIB8g"])


# In[12]:


#Creating Y values
y_map={}
for i in range(len(reader)):
    if(reader["elite"][i]=="None"):
        y_map[reader["user_id"][i]]=0
    else:
        y_map[reader["user_id"][i]]=1


# In[13]:


#creating feature vector
feature=[]
count=0
for key in f:
    try:
        temp=[]
        temp.append(key)
        temp.append(f[key]["review_count"])
        temp.append(f[key]["yelping_age"])
        temp.append(f[key]["avg_rating"])
        temp.append(f[key]["stdev"])
        temp.append(f[key]["unique_business_count"])
        temp.append(y_map[key])
        feature.append(temp)
        if(y_map[key]==0):
            count+=1
    except:
        print(key)
print(count)


# In[14]:


#print(len(feature))
feature_df=pd.DataFrame(feature,columns=["user_id","review_count","yelping_age","avg_rating","stdev","unique_business_count","y"])


# In[15]:


print(reader['elite'][0]=="None")
print(len(feature))


# In[16]:


feature_df.to_csv("F://topical_features.txt",index=False)
