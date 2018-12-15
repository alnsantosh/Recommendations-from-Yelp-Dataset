
# coding: utf-8

# In[1]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier

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


reader=pd.read_csv('F://user_location.csv',delimiter=",",header=None)
reader.columns=["user_id","latitude","longitude"]


# In[3]:


reader.head()


# In[4]:


len(reader)


# In[5]:


#Handersine method for calculating the distance between two coordinates in the globe
def calculate_distance(latitude1,longitude1,latitude2,longitude2):
    latitude1,longitude1,latitude2,longitude2=map(math.radians,[latitude1,longitude1,latitude2,longitude2])#Converting degrees to radians
    net_latitude=latitude2-latitude1
    net_longitude=longitude2-longitude1
    temp=math.sin(net_latitude/2)**2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(net_longitude/2)**2
    return 3956*2*math.asin(math.sqrt(temp))


# In[6]:


calculate_distance(33.461890,-112.069980,33.461890,-112.069980)



def find_elite(latitude,longitude):
    user_dist=[]
    e=""
    for index,row in reader.iterrows():
        user=[]
        temp=calculate_distance(reader["latitude"][index],reader["longitude"][index],latitude,longitude)
        e=reader["user_id"][index]
        if(temp<=15):
            user.append(e)
            user.append(temp)
            user_dist.append(user);
    return user_dist


# In[19]:


dist=find_elite(51.0918130155,-114.031674872)


# In[20]:


print(len(dist))


# In[21]:


dist[:10]


# In[22]:


dist = sorted(dist, key = lambda x:x[1])#Sorting based on the distance between user and elite users
print(dist[:5]) # Gives list of five elite users for a input location

