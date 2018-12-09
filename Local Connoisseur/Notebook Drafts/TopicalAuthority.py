
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


# In[107]:


from collections import defaultdict
import math
import pickle


# In[3]:


reader=pd.read_csv('F://topical.txt',delimiter="\t",header=None,names=["business_id","business_stars","business_review_count","review_id","review_stars","text","user_id","name","user_review_count","yelping_since","friends","useful","funny","cool","fans","elite"],encoding='latin-1')


# In[105]:


reader.head()


# In[4]:


f=defaultdict(dict)
for i in range(len(reader)):
    f[reader["user_id"][i]]={};
    


# In[5]:


print(f[reader["user_id"][2]])
print(len(f))
print(len(reader))
#reader=reader[:5]


# In[6]:


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
    
        


# In[7]:


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
    


# In[8]:


# print(f["-3i9bhfvrM3F1wsC9XIB8g"])
# print(f["-3i9bhfvrM3F1wsC9XIB8g"]["ratings"])
# print(len(remove_keys))

for key in remove_keys:
    f.pop(key,None)


# In[9]:


# print(type(reader["yelping_since"][39721]))
print(f["-3i9bhfvrM3F1wsC9XIB8g"])


# In[10]:


#Creating Y values
y_map={}
for i in range(len(reader)):
    if(reader["elite"][i]=="None"):
        y_map[reader["user_id"][i]]=0
    else:
        y_map[reader["user_id"][i]]=1


# In[11]:


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


# In[54]:


#print(len(feature))
feature_df=pd.DataFrame(feature,columns=["user_id","review_count","yelping_age","avg_rating","stdev","unique_business_count","y"])


# In[55]:


print(reader['elite'][0]=="None")
print(len(feature))


# In[56]:


feature_df['is_train']=np.random.uniform(0,1,len(feature_df)) <=0.75
#reader.head()


# In[57]:


train=feature_df[feature_df['is_train']==True]
test=feature_df[feature_df['is_train']==False]
print(len(train))
print(len(test))
print(len(train)/(len(train)+len(test)))


# In[58]:


features=feature_df.columns[1:-2]
features
#print(train[features])
y=train["y"]
#print(y)


# In[59]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None)
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
    


# In[62]:


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


# In[63]:


precision=tp/(tp+fp)
recall=tp/(tp+fn)
f_measure=2*(precision*recall)/(precision+recall)
print(precision)
print(recall)
print(f_measure)


# In[47]:


#To remove the skewness, making the number of positive and negative examples as same


# In[64]:


feature_df.head()


# In[65]:


feature_df_new=feature_df[:]


# In[88]:


feature_df_new=feature_df_new.drop(labels="is_train",axis=1)


# In[96]:


print(len(feature_df_new[feature_df_new['y']==1]))
print(len(feature_df_new[feature_df_new['y']==0]))
feature_1=feature_df_new[feature_df_new['y']==1]
feature_0=feature_df_new[feature_df_new['y']==0][:50000]
print(len(feature_1))
print(len(feature_0))


# In[97]:


feature_nonskew=feature_1.append(feature_0)


# In[98]:


print(len(feature_0))
print(len(feature_1))
print(len(feature_nonskew))
feature_nonskew.head()


# In[99]:


#RandomForest on Non-Skewed data
feature_nonskew['is_train']=np.random.uniform(0,1,len(feature_nonskew)) <=0.75
train=feature_nonskew[feature_nonskew['is_train']==True]
test=feature_nonskew[feature_nonskew['is_train']==False]
print(len(train))
print(len(test))
print(len(train)/(len(train)+len(test)))

features=feature_nonskew.columns[1:-2]
features
#print(train[features])
y=train["y"]
#print(y)


# In[100]:


clf=RandomForestClassifier(random_state=0,n_estimators=150,max_depth=None)
clf.fit(train[features],y)


# In[101]:


test_y=clf.predict(test[features])


# In[102]:


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


# In[109]:


with open("F://DM_Model_RF/model.pkl","wb") as f:
    pickle.dump(clf,f)


# In[114]:


feature_nonskew.to_csv("F://DM_DataFrames/feature.csv",index=False)


# In[115]:


feature_df.to_csv("F://DM_DataFrames/feature_big.csv",index=False)


# In[112]:


len(feature_df)


# In[104]:


#from sklearn.mixture import GMM


# In[ ]:


# X=
# gmm=GMM(n_components=4).fit(X)

