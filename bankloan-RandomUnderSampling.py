#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


bank=pd.read_csv(r"bank_marketing.csv",header=2)


# In[4]:


bank.head()


# In[5]:


cred=bank[['jobedu','month']]


# In[6]:


tmp = cred.stack().str.split(',', expand=True).unstack(level=1).sort_index(level=1, axis=1)
tmp.columns = [f'{y}_{x+1}' for x, y in tmp.columns]
out = cred.join(tmp).dropna(how='all', axis=1).fillna('')


# In[7]:


out.head()


# In[8]:


out=out[['jobedu_1','jobedu_2','month_1','month_2']]


# In[9]:


out.head()


# In[10]:


out=out.rename(columns={'jobedu_1':'job','jobedu_2':'education','month_1':'month','month_2':'year'})


# In[11]:


out.columns


# In[12]:


bank.drop(columns=['jobedu','month'],inplace=True)


# In[13]:


bank.head()


# In[14]:


bank=pd.concat([bank,out],axis=1)


# In[15]:


bank.columns


# In[16]:


bank.dtypes


# In[17]:


dur=bank[['duration']]


# In[18]:


temp = dur.stack().str.split(' ', expand=True).unstack(level=1).sort_index(level=1, axis=1)
temp.columns = [f'{y}_{x+1}' for x, y in temp.columns]
durout = dur.join(temp).dropna(how='all', axis=1).fillna('')


# In[19]:


durout.head()


# In[20]:


durout=durout[['duration_1','duration_2']]


# In[21]:


durout.iloc[15182]


# In[22]:


durout['duration_1']=pd.to_numeric(durout['duration_1'],errors="coerce")


# In[23]:


type(durout)


# In[24]:


durout['duration_1']=durout['duration_1'].astype(int)


# In[25]:


durout.iloc[15182]


# In[26]:


durout['duration_2'] = durout['duration_2'].replace(['sec','min'],[1,60])
        


# In[27]:


durout.head(5)


# In[28]:


durout['duration_2']=pd.to_numeric(durout['duration_2'],errors="coerce")


# In[29]:


durout.dtypes


# In[30]:


durout['duration_in_min']=durout['duration_1']*durout['duration_2']


# In[31]:


durout.drop(['duration_1', 'duration_2'],axis=1,inplace=True)


# In[32]:


bank=pd.concat([bank,durout],axis=1)


# In[33]:


bank.columns


# In[34]:


bank.drop(['duration'],axis=1,inplace=True)


# In[35]:


bank.head()


# In[36]:


bank.marital.value_counts()


# In[37]:


bank['marital']=bank['marital'].astype(str)


# In[38]:


# Data Imbalance

bank.marital.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["married","single","divorced"])
plt.ylabel("")
plt.show()


# In[39]:


bank.targeted.value_counts()


# In[40]:


bank['targeted']=bank['targeted'].astype(str)


# In[41]:


bank.targeted.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["yes","no"])
plt.ylabel("")
plt.show()


# In[42]:


bank.default.value_counts()


# In[43]:


bank['default']=bank['default'].astype(str)


# In[44]:


bank.default.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["no","yes"])
plt.ylabel("")
plt.show()


# In[45]:


bank.loan.value_counts()


# In[46]:


bank['loan']=bank['loan'].astype(str)


# In[47]:


bank.loan.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["no","yes"])
plt.ylabel("")
plt.show()


# In[48]:


bank.contact.value_counts()


# In[49]:


bank['contact']=bank['contact'].astype(str)


# In[50]:


bank.housing.value_counts()


# In[51]:


bank.housing.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["no","yes"])
plt.ylabel("")
plt.show()


# In[52]:


bank.poutcome.value_counts()


# In[53]:


bank['poutcome']=bank['poutcome'].astype(str)


# In[54]:


bank.poutcome.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["unknown","failure","other","success"])
plt.ylabel("")
plt.show()


# In[55]:


bank.response.value_counts()


# In[56]:


bank.response.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["no","yes"])
plt.ylabel("")
plt.show()


# In[57]:


bank['response']=bank['response'].astype(str)


# In[58]:


bank.job.value_counts()


# In[59]:


bank['job']=bank['job'].astype(str)


# In[60]:


bank.education.value_counts()


# In[61]:


bank['education']=bank['education'].astype(str)


# In[62]:


bank.education.value_counts(normalize=True).plot.pie(autopct = "%1.2f%%",labels=["primary","secondary","tertiary","unknown"])
plt.ylabel("")
plt.show()


# In[63]:


bank.month.value_counts()


# In[64]:


bank['month']=bank['month'].astype(str)


# In[65]:


bank.month.value_counts()


# In[66]:


bank.year.value_counts()


# In[67]:


bank=bank.dropna()


# In[68]:


bank.isna().any().sum()


# In[69]:


bank.dtypes


# In[70]:


for i in bank.columns:
    if bank[i].dtype=='object':
        print(bank[i].value_counts())


# In[71]:


# for col in ['targeted','default','housing','loan','response','marital','contact','poutcome','job','education']:
#     bank[col] = bank[col].astype('category')


# In[72]:


bank.dtypes


# In[73]:


bank.drop(['customerid','year'],axis=1,inplace=True)


# In[74]:


bank['age']=np.int64(bank['age'])


# In[75]:


bank.dtypes


# In[76]:


bank.age.value_counts()


# In[77]:


bank['response']=bank['response'].replace(['yes','no'],[1,0])
bank['targeted']=bank['targeted'].replace(['yes','no'],[1,0])
bank['default']=bank['default'].replace(['yes','no'],[1,0])
bank['housing']=bank['housing'].replace(['yes','no'],[1,0])
bank['loan']=bank['loan'].replace(['yes','no'],[1,0])
bank['pdays']=bank['pdays'].replace([-1],[0])


# In[78]:


bank.columns


# In[79]:


cols=['targeted', 'default', 'housing','loan']

for col in cols:
    bank[col]=bank[col].apply(str)


# In[80]:


bank=bank.dropna()


# In[81]:


bank.index[bank['response'] == np.nan].tolist()


# In[82]:


bank.response.value_counts()


# In[83]:


#bank=bank.reset_index(drop=True)


# In[84]:


#bank = bank[bank['response'].notna()]


# In[85]:


bank = bank[bank.response != 'nan']


# In[86]:


bank.response.value_counts()


# In[87]:


bank['response']=pd.to_numeric(bank['response'])


# In[88]:


bank.poutcome.value_counts()


# #### Handling Missing Values

# In[89]:


bank=bank.dropna()


# In[90]:


bank.isnull().sum().sum()


# #### Imputation

# In[91]:


# Calculating the percentage of 'unknown' in 'contact' column and impute it.
bank['contact'].value_counts()[1]/len(bank['contact'])


# In[92]:


bank['contact'].value_counts()


# In[93]:


bank['contact']=bank['contact'].replace(['unknown'],'cellular')


# In[94]:


bank['contact'].value_counts()


# In[95]:


bank['poutcome'].value_counts()


# In[96]:


bank['job'].value_counts()


# In[97]:


bank.job.mode()


# In[98]:


bank['job']=bank['job'].replace(['unknown'],bank.job.mode())


# In[99]:


bank['job'].value_counts()


# In[100]:


bank.education.value_counts()


# In[101]:


bank.education.value_counts()[3]/len(bank.education)


# In[102]:


bank['education']=bank['education'].replace(['unknown'],bank.education.mode())


# In[103]:


bank.pdays.value_counts()


# In[104]:


len(bank.pdays)


# In[105]:


bank.dtypes


# In[106]:


len(bank.targeted)


# In[107]:


bank.targeted.value_counts()


# In[108]:


numeric=[]
for col in bank.columns:
    if bank[col].dtype==np.int64:
        numeric.append(col)


# In[109]:


bank_num=bank[numeric]


# In[110]:


bank_num.head()


# In[111]:


plt.boxplot(bank_num.age)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[112]:


plt.boxplot(bank_num.salary)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[113]:


plt.boxplot(bank_num.balance)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[114]:


plt.boxplot(bank_num.day)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[115]:


plt.boxplot(bank_num.campaign)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[116]:


plt.boxplot(bank_num.duration_in_min)
fig = plt.figure(figsize =(10, 7))
plt.show()


# In[117]:


q_low = bank_num["balance"].quantile(0.08) 
q_hi = bank_num["balance"].quantile(0.92) 

bank_num = bank_num[(bank_num["balance"] < q_hi) & (bank_num["balance"] > q_low)]


# In[118]:


q_low_camp = bank_num["campaign"].quantile(0.08) 
q_hi_camp = bank_num["campaign"].quantile(0.92) 

bank_num = bank_num[(bank_num["campaign"] < q_hi_camp) & (bank_num["campaign"] > q_low_camp)]


# In[119]:


q_low_dur = bank_num["duration_in_min"].quantile(0.08) 
q_hi_dur = bank_num["duration_in_min"].quantile(0.92) 

bank_num = bank_num[(bank_num["duration_in_min"] < q_hi_dur) & (bank_num["duration_in_min"] > q_low_dur)]


# In[120]:


bank_num.shape


# In[121]:


bank.shape


# In[122]:


non_numeric=[]
for col in bank.columns:
    if bank[col].dtype==object:
        non_numeric.append(col)


# In[123]:


bank_obj=bank[non_numeric]


# In[124]:


bank_obj.columns


# In[125]:


x=pd.concat([bank_obj,bank_num],axis=1)


# In[126]:


x.isna().sum()


# In[127]:


x.columns


# In[128]:


x.shape


# In[129]:


# Imputation by mean

(x.age.isna().sum()/x.shape[0])*100


# In[130]:


x.age.fillna(value=int(x['age'].mean()),inplace=True)


# In[131]:


(x.salary.isna().sum()/x.shape[0])*100


# In[132]:


x.salary.fillna(value=int(x['salary'].mean()),inplace=True)


# In[133]:


(x.balance.isna().sum()/x.shape[0])*100


# In[134]:


x.balance.fillna(value=int(x['balance'].mean()),inplace=True)


# In[135]:


x.balance.mean()


# In[136]:


# As all the numeric columns are having same no. of null values we simply impute with mean values
x.day.fillna(value=int(x['day'].mean()),inplace=True)


# In[137]:


x.campaign.fillna(value=int(x['campaign'].mean()),inplace=True)


# In[138]:


x.pdays.fillna(value=int(x['pdays'].mean()),inplace=True)


# In[139]:


x.previous.fillna(value=int(x['previous'].mean()),inplace=True)


# In[140]:


int(x['response'].mean())


# In[141]:


x.response.fillna(value=int(x['response'].mode()),inplace=True)


# In[142]:


x.response.value_counts()


# In[143]:


x.duration_in_min.fillna(value=int(x['duration_in_min'].mean()),inplace=True)


# In[144]:


x.columns


# In[145]:


y=x['response']
x=x.drop(columns=['response'],axis=1)


# In[146]:


x.shape


# In[147]:


y.shape


# In[148]:


non_numeric=[]
for col in x.columns:
    if x[col].dtype==object:
        non_numeric.append(col)


# In[149]:


x_obj=x[non_numeric]


# In[150]:


x_obj.columns


# ### Dummy Variables

# In[151]:


bank_dummy=pd.get_dummies(data=x_obj,columns=x_obj.columns,drop_first=True)


# In[152]:


bank_dummy.head()


# In[153]:


x.columns


# In[154]:


bank_no_object=x.drop(columns=x_obj.columns,axis=1)


# In[155]:


X=pd.concat([bank_no_object,bank_dummy],axis=1)


# In[156]:


X.columns


# ### Dealing with Imbalanced Data

# In[157]:


# import library
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)

# fit predictor and target variable

x_rus, y_rus = rus.fit_resample(X, y)


# ### Feature Selection

# In[158]:


from sklearn.ensemble import RandomForestClassifier


# In[159]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[160]:


# X=bank.iloc[:, :-1]
# y=bank.iloc[:,-1]


# In[161]:


x_rus.head()


# In[162]:


y_rus.head()


# In[163]:


x_rus.shape


# In[164]:


y_rus.shape


# In[165]:


X_train,X_test,y_train,y_test=train_test_split(x_rus,y_rus,test_size=0.3,random_state=1121218)


# In[166]:


X_train.shape


# In[167]:


X_test.shape


# In[168]:


y_train.shape


# In[169]:


y_test.shape


# #### Feature Selection

# In[170]:


from sklearn.feature_selection import RFE


# In[171]:


rfe=RFE(estimator=RandomForestClassifier(),n_features_to_select=10)


# In[172]:


_ =rfe.fit(X_train,y_train)


# In[173]:


X_train=X_train.loc[:,rfe.support_]


# In[174]:


X_test=X_test.loc[:,rfe.support_]


# In[175]:


X_train.head()


# In[176]:


X_test.head()


# In[177]:


sc=StandardScaler()


# In[178]:


X_train=sc.fit_transform(X_train)


# In[179]:


X_test=sc.transform(X_test)


# In[180]:


y_train=np.array(y_train).reshape(-1,1)
# y_train = y_train_sc.fit_transform(y_train_sc)


# In[181]:


y_test=np.array(y_test).reshape(-1,1)
# y_test = y_test_sc.transform(y_test_sc)


# In[182]:


# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
      '''Function to perform 5 Folds Cross-Validation
       Parameters
       ----------
      model: Python Class, default=None
              This is the machine learning algorithm to be used for training.
      _X: array
           This is the matrix of features.
      _y: array
           This is the target variable.
      _cv: int, default=5
          Determines the number of folds for cross-validation.
       Returns
       -------
       The function returns a dictionary containing the metrics 'accuracy', 'precision',
       'recall', 'f1' for both training set and validation set.
      '''
      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }


# ### Model Building

# In[183]:


from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


# #### Support Vector

# In[184]:


from sklearn.svm import SVC

svc_classifier=SVC()
svc_classifier.fit(X_train,y_train)


# In[185]:


y_pred_svc=svc_classifier.predict(X_test)


# In[186]:


svc_accuracy=round((accuracy_score(y_test,y_pred_svc)*100),2)


# In[187]:


svc_accuracy


# In[188]:


# cross validation

svc_result = cross_validation(svc_classifier, x_rus, y_rus, 5)
print(svc_result)


# In[ ]:





# #### Logisitic Regression

# In[189]:


from sklearn.linear_model import LogisticRegression


# In[190]:


y_train.shape


# In[191]:


y_test.shape


# In[192]:


lr_classifier=LogisticRegression()
lr_classifier.fit(X_train,y_train)


# In[193]:


y_pred_lr=lr_classifier.predict(X_test)


# In[194]:


lr_accuracy=round(accuracy_score(y_test,y_pred_lr)*100,2)


# In[195]:


lr_accuracy


# In[196]:


# cross validation

lr_result = cross_validation(lr_classifier, x_rus, y_rus, 5)
print(lr_result)


# In[ ]:





# #### K-Nearest Neighbour Classifier

# In[197]:


from sklearn.neighbors import KNeighborsClassifier


# In[198]:


knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
knn_classifier.fit(X_train,y_train)


# In[199]:


y_pred_knn=knn_classifier.predict(X_test)
knn_accuracy=round(accuracy_score(y_test,y_pred_knn)*100,2)


# In[200]:


knn_accuracy


# In[201]:


# cross validation

knn_result = cross_validation(knn_classifier, x_rus, y_rus, 5)
print(knn_result)


# #### Naive Bayes Classifier

# In[202]:


from sklearn.naive_bayes import GaussianNB


# In[203]:


nb_classifier=GaussianNB()
nb_classifier.fit(X_train,y_train)


# In[204]:


y_pred_nb=nb_classifier.predict(X_test)
nb_accuracy=round(accuracy_score(y_test,y_pred_nb)*100,2)


# In[205]:


nb_accuracy


# In[206]:


# cross validation

nb_result = cross_validation(nb_classifier, x_rus, y_rus, 5)
print(nb_result)


# #### Decision Classifier

# In[207]:


from sklearn.tree import DecisionTreeClassifier


# In[208]:


dt_classifier=DecisionTreeClassifier()
dt_classifier.fit(X_train,y_train)


# In[209]:


y_pred_dt=dt_classifier.predict(X_test)
dt_accuracy=round(accuracy_score(y_test,y_pred_dt)*100,2)


# In[210]:


dt_accuracy


# In[211]:


# cross validation

dt_result = cross_validation(dt_classifier, x_rus, y_rus, 5)
print(dt_result)


# #### Random Forest Classifier

# In[212]:


from sklearn.ensemble import RandomForestClassifier


# In[213]:


rfc_classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=51)
rfc_classifier.fit(X_train,y_train)


# In[214]:


y_pred_rfc=rfc_classifier.predict(X_test)
rfc_accuracy=round(accuracy_score(y_test,y_pred_rfc)*100,2)


# In[215]:


rfc_accuracy


# In[216]:


# cross validation

rfc_result = cross_validation(rfc_classifier, x_rus, y_rus, 5)
print(rfc_result)


# #### Ada Boost Classifier

# In[217]:


from sklearn.ensemble import AdaBoostClassifier


# In[218]:


adb_classifier=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',random_state=21),
                                 n_estimators=2000,
                                 learning_rate=0.1,
                                 algorithm='SAMME.R',
                                 random_state=1)
adb_classifier.fit(X_train,y_train)


# In[219]:


y_pred_adb=adb_classifier.predict(X_test)
adb_accuracy=round(accuracy_score(y_test,y_pred_adb)*100,2)


# In[220]:


adb_accuracy


# In[221]:


# Cross validation

adb_result = cross_validation(adb_classifier, x_rus, y_rus, 5)
print(adb_result)


# #### XGBoost Classifier

# In[222]:


import xgboost as xgb


# In[223]:


xgb_classifier=xgb.XGBClassifier()
#xgb_classifier.fit(X_train._get_numeric_data(),y_train)
xgb_classifier.fit(X_train,y_train)


# In[224]:


#y_pred_xgb=xgb_classifier.predict(X_test._get_numeric_data())
y_pred_xgb=xgb_classifier.predict(X_test)
xgb_accuracy=round(accuracy_score(y_test,y_pred_xgb)*100,2)


# In[225]:


xgb_accuracy


# In[226]:


# Cross validation

xgb_result = cross_validation(xgb_classifier, x_rus, y_rus, 5)
print(xgb_result)


# In[227]:


ultimate_model={'xgb_classifier':'xgb_classifier','nb_classifier':'nb_classifier',
                'adb_classifier':'adb_classifier',
                'knn_classifier':'knn_classifier',
                'dt_classifier':'dt_classifier',
                'lr_classifier':'lr_classifier',
                'rfc_classifier':'rfc_classifier'}


# In[228]:


ultimate_classifier={'xgb_classifier':xgb_accuracy,'nb_classifier':nb_accuracy,'adb_classifier':adb_accuracy,
                'knn_classifier':knn_accuracy,'dt_classifier':dt_accuracy,'lr_classifier':lr_accuracy,
                'svc_classifier':svc_accuracy,
                'rfc_classifier':rfc_accuracy}


# In[229]:


keymax=max(zip(ultimate_classifier.values(),ultimate_classifier.keys()))[1]


# In[230]:


keymax


# In[231]:


list(ultimate_classifier.keys())


# In[232]:


list(ultimate_classifier.values())


# In[233]:


final_model=list(ultimate_model.keys())[list(ultimate_model.values()).index(keymax)]


# ### Saving the Model

# In[234]:


import pickle

#dump information to that file
pickle.dump(final_model,open('model.pkl','wb'))

#load a model
pickle.load(open('model.pkl','rb'))


# In[ ]:





# In[ ]:




