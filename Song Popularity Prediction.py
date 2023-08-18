#!/usr/bin/env python
# coding: utf-8

# In[140]:


#Importing required libraries and loading the dataset
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df=pd.read_csv('C:/Users/Satyaanalamadhusamhi/Desktop/song_data.csv')


# In[141]:


df.head()


# In[142]:


df.shape


# In[143]:


#Remove useless columns
df.drop('song_name',axis=1,inplace=True)


# In[144]:


df.head()


# In[145]:


df.shape


# In[146]:


#Remove rows with null values
df.dropna()


# In[147]:


#Remove duplicate rows
df.drop_duplicates(inplace=True)


# In[148]:


df.shape


# In[149]:


df.info()


# In[150]:


#prepare the data
y=df['song_popularity']
x=df.drop('song_popularity',axis=1)
target = 'song_popularity'
features = [i for i in df.columns if i not in [target]]


# In[151]:


y.head()


# In[152]:


x.head()


# In[153]:


#Checking no.of unique rows in each feature
df.nunique().sort_values()


# In[154]:


#Checking number of unique rows in each feature

nu = df[features].nunique().sort_values()
nf = []; cf = []; #numerical & categorical features

for i in range(df[features].shape[1]):
    if nu.values[i]<=8:cf.append(nu.index[i])
    else: nf.append(nu.index[i])

print(len(cf),len(nf))
cf


# In[ ]:





# In[155]:


#Check statistics of all features
df.describe()


# In[156]:


#EDA
#Distribution of target variable
plt.figure(figsize=[8,4])
sns.distplot(df[target], color='g',hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution - Median Value of Homes ($1Ms)')
plt.show()


# In[157]:


#Visualising the categorical features 

print('\033[1mVisualising Categorical Features:'.center(100))

n=2
plt.figure(figsize=[15,3*math.ceil(len(cf)/n)])

for i in range(len(cf)):
    if df[cf[i]].nunique()<=8:
        plt.subplot(math.ceil(len(cf)/n),n,i+1)
        sns.countplot(df[cf[i]])
    else:
        plt.subplot(2,1,2)
        sns.countplot(df[cf[i]])
        
plt.tight_layout()
plt.show()


# In[158]:


#Visualising the numeric features 

print('\033[1mNumeric Features Distribution'.center(100))

n=5

clr=['r','g','b','g','b','r']

plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    sns.distplot(df[nf[i]],hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255,255,255])/255))
plt.tight_layout()
plt.show()

plt.figure(figsize=[15,4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3),n,i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()


# In[159]:


#Removal of outlier:

df1 = df.copy()

features1 = nf

for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
display(df1.head())
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(df.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))


# In[160]:


df1.shape


# In[161]:


X = df1.drop([target],axis=1)
Y = df1[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True,inplace=True)

print('Original set  ---> ',X.shape,Y.shape,'\nTraining set  ---> ',Train_X.shape,Train_Y.shape,'\nTesting set   ---> ', Test_X.shape,'', Test_Y.shape)


# In[ ]:





# In[162]:


#Feature Scaling (Standardization)

std = StandardScaler()

print('\033[1mStandardardization on Training set'.center(120))
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
display(Train_X_std.describe())

print('\n','\033[1mStandardardization on Testing set'.center(120))
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
display(Test_X_std.describe())


# In[163]:


#Checking the correlation

print('\033[1mCorrelation Matrix'.center(100))
plt.figure(figsize=[10,8])
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, center=0,cmap='coolwarm')
plt.show()


# In[170]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(Train_X,Train_Y)
acc=lr.score(Test_X,Test_Y)
print(f'Linear regression:{acc*100}')


# In[171]:


Pred_Y=lr.predict(Test_X)


# In[172]:


Pred_Y


# In[182]:


print("Comparision of original test data and predicted values")
df2=pd.DataFrame({'Actual_values':Test_Y,'Predicted_values':Pred_Y})
print(df2)
print('\n\n')


# In[173]:


from sklearn.metrics import mean_squared_error
mean_squared_error(Test_Y,Pred_Y)


# In[176]:


rmse=math.sqrt(mean_squared_error(Test_Y,Pred_Y))
rmse


# In[178]:


from sklearn.metrics import r2_score
r=r2_score(Test_Y,Pred_Y)
r          


# In[179]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Test_Y,Pred_Y)


# In[180]:


#Evaluate the performance
from sklearn.metrics import classification_report
report=classification_report(Test_Y,Pred_Y)
print("Classification Report")
print(report)


# In[ ]:




