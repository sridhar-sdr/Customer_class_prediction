
# coding: utf-8

# # Great_Customer_prediction

# ## Importing Necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings

warnings.filterwarnings('ignore')


# ## Read data

# In[4]:


data=pd.read_csv('https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv')
data


# # Checking data Imbalance

# In[5]:


data.great_customer_class.value_counts().plot.pie(autopct='%.2f')


# Data is heavily imbalanced

# When working with imbalanced data, we don’t recommend using categorical accuracy as the main evaluation measure. It is not unusual to observe a high evaluation accuracy when testing a classification model trained on very imbalanced data. 

# # Performance Metric

# Precision/Specificity: how many selected instances are relevant.
#     
# Recall/Sensitivity: how many relevant instances are selected.
#     
# F1 score: harmonic mean of precision and recall.
#     
# AUC: relation between true-positive rate and false positive rate
# 
# Confusion Matrix

# In[6]:


data.head(5)


# # Shape

# In[7]:


data.shape


# # Features

# In[8]:


data.columns


# # check the missing values

# In[9]:


data.isna().sum()


# In[10]:


data.info()


# ## Checking the missing value percentage

# In[11]:


for i in range(len(data.columns)):
    missing_data = data[data.columns[i]].isna().sum()
    perc = missing_data / len(data) * 100
    print(f'Feature {i+1} >> Missing entries: {missing_data}  |  Percentage: {round(perc, 2)}')


# # Visualizing the missing values

# In[12]:


plt.figure(figsize=(10,6))
sns.heatmap(data.isna(), cbar=False, cmap='viridis', yticklabels=False)


# ## Handling categorical Missing Values

# In[13]:


data['workclass']= data['workclass'].fillna('U')
data['occupation']= data['occupation'].fillna('U')


# In[14]:


data.isnull().sum()


# ## Handling numerical missing values

# In[15]:


from numpy import NaN
data[['age','salary','mins_beerdrinking_year','mins_exercising_year','tea_per_year','coffee_per_year']] = data[['age','salary','mins_beerdrinking_year','mins_exercising_year','tea_per_year','coffee_per_year']].replace(0, NaN)


# In[16]:


data.fillna(data.mean(), inplace=True)
data


# ## Zero Missing Values

# In[17]:


data.isnull().sum()


# ## Handling Categorical data

# In[18]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm=data.apply(LabelEncoder().fit_transform)
xm


# # Input features

# In[20]:


X=xm.iloc[:,:-1]
X


# ## Output Features

# In[21]:


y=xm.iloc[:,7]
y


# ## Feature importance

# In[22]:


#importing the ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


# In[23]:


model = ExtraTreesClassifier()


# In[24]:


model.fit(X,y)

print(model.feature_importances_)


# In[25]:


#Top 5 important features
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# In[26]:


#X.columns
X_new = X[['occupation','user_id','salary','workclass','race']]
X_new


# # Split data

# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.30, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[28]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_train_over, y_train_over = oversample.fit_resample(X,y)

# summarize class distribution
print("After oversampling: ",Counter(y_train_over))

Undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_train_under, y_train_under = Undersample.fit_resample(X,y)

# summarize class distribution
print("After Undersampling: ",Counter(y_train_under))
X_combined_sampling, y_combined_sampling = Undersample.fit_resample(X_train_over, y_train_over)
print(f"Combined Random Sampling: {Counter(y_combined_sampling)}")


# # Model Building

# # KNN

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
print(knn_classifier.score(X_test, y_test))
print(knn_classifier.score(X_train, y_train))


# In[30]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,knn_predictions)


# In[35]:



knn_classifier_n = KNeighborsClassifier()
knn_classifier_n.fit(X_train_over, y_train_over)


# In[38]:


knn_predictions1 = knn_classifier_n.predict(X_test)
print(knn_classifier_n.score(X_test, y_test))
print(knn_classifier_n.score(X_train_over, y_train_over))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,knn_predictions1)


# In[39]:


model1=knn_classifier.fit(X_train_under, y_train_under)


# In[40]:


knn_predictions_new = knn_classifier.predict(X_test)
print(knn_classifier.score(X_test, y_test))
print(knn_classifier.score(X_train_under, y_train_under))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,knn_predictions_new)


# In[ ]:


knn_classifier.fit(X_combined_sampling, y_combined_sampling)


# In[ ]:


knn_predictions_new1 = knn_classifier.predict(X_test)
print(knn_classifier.score(X_test, y_test))
print(knn_classifier.score(X_combined_sampling, y_combined_sampling))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,knn_predictions_new1)


# In[ ]:


from sklearn.model_selection import GridSearchCV
grid_params= {'n_neighbors':[3,5,11,19],'weights':['uniform','distance'],'metric':['euclidean','manhattan']
             }


# In[ ]:


gridsearch= GridSearchCV(knn_classifier,grid_params, verbose=1,cv=3,n_jobs=-1)


# In[ ]:


gs_results_knn=gridsearch.fit(X_combined_sampling, y_combined_sampling)


# In[ ]:


print(gs_results_knn.best_score_)
print(knn_classifier.score(X_combined_sampling, y_combined_sampling))


# In[ ]:


gs_results_knn.best_estimator_


# In[ ]:


gs_results_knn.best_params_


# In[ ]:


knn_pred=knn_classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test_new,knn_pred)


# # Random Forest

# In[41]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()

rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
print(rf_classifier.score(X_test, y_test))
print(rf_classifier.score(X_train, y_train))


# In[ ]:


#overfitting


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,rf_predictions)


# In[ ]:


#RandomForest with randomSearchCv


# In[ ]:


RSEED=50
rf = RandomForestClassifier(random_state= RSEED)
from pprint import pprint
# Look at parameters used by our current forest

#print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']+list(np.arange(0.5, 1, 0.1))
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 20, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=RSEED,n_jobs = -1)


# In[ ]:


rf_random.fit(X_train, y_train)


# In[ ]:


print(rf_random.best_params_)


# In[ ]:


random_cv=rf_random.best_estimator_
random_cv


# In[ ]:


y_pred1 = random_cv.predict(X_test)
y_pred1


# In[ ]:


print(random_cv.score(X_test,y_test))
print(random_cv.score(X_train,y_train))


# #RandomForest with Oversampling

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_train_over, y_train_over = oversample.fit_resample(X,y)

# summarize class distribution
print("After oversampling: ",Counter(y_train_over))

Undersample = RandomUnderSampler(sampling_strategy='majority')
# fit and apply the transform
X_train_under, y_train_under = Undersample.fit_resample(X,y)

# summarize class distribution
print("After oversampling: ",Counter(y_train_under))
X_combined_sampling, y_combined_sampling = Undersample.fit_resample(X_train_over, y_train_over)
print(f"Combined Random Sampling: {Counter(y_combined_sampling)}")


# In[ ]:


oversample = RandomOverSampler(sampling_strategy='minority')


# In[ ]:


# fit and apply the transform
X_train_over, y_train_over = oversample.fit_resample(X,y)

# summarize class distribution
print("After oversampling: ",Counter(y_train_over))


# In[ ]:


rf_random.fit(X_train_over, y_train_over)


# In[ ]:


random_cv_new1=rf_random.best_estimator_
random_cv_new1


# In[ ]:


print(random_cv_new1.score(X_test,y_test))
print(random_cv_new1.score(X_train_over,y_train_over))


# In[ ]:


Undersample = RandomUnderSampler(sampling_strategy='majority')


# In[ ]:


# fit and apply the transform
X_train_under, y_train_under = Undersample.fit_resample(X,y)

# summarize class distribution
print("After oversampling: ",Counter(y_train_under))


# In[ ]:


rf_random.fit(X_train_under, y_train_under)


# In[ ]:


random_cv_new=rf_random.best_estimator_
random_cv_new


# In[ ]:


print(random_cv_new.score(X_test,y_test))
print(random_cv_new.score(X_train_under,y_train_under))


# In[ ]:


X_combined_sampling, y_combined_sampling = Undersample.fit_resample(X_train_over, y_train_over)
print(f"Combined Random Sampling: {Counter(y_combined_sampling)}")


# In[ ]:


ax=y_combined_sampling.value_counts().plot.pie(autopct='%.2f')


# In[ ]:


rf_random.fit(X_combined_sampling, y_combined_sampling)


# In[ ]:


random_cv_combined=rf_random.best_estimator_
random_cv_combined


# In[ ]:


print(random_cv_combined.score(X_test,y_test))
print(random_cv_combined.score(X_combined_sampling, y_combined_sampling))


# # Logistic Regression

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state=5)


model = LogisticRegression()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.score(X_train, y_train))
lr_predictions=model.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,lr_predictions)


# In[ ]:


combined_model = LogisticRegression()
combined_model.fit(X_train_under, y_train_under)


# In[ ]:


print(combined_model.score(X_test, y_test))
print(combined_model.score(X_train_under, y_train_under))


# In[ ]:


from sklearn.metrics import confusion_matrix
lr_predictions_new=model.predict(X_test)
confusion_matrix(y_test,lr_predictions_new)


# In[ ]:


from sklearn.model_selection import RepeatedStratifiedKFold
# Create grid search object
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
lg = grid_search.fit(X_combined_sampling, y_combined_sampling)


# In[ ]:


lg.best_score_


# # SVM

# In[42]:


from sklearn import svm
Sv_Classifier= svm.SVC()


# In[43]:


Sv_Classifier.fit(X_combined_sampling, y_combined_sampling)


# In[44]:


Sv_predictions = Sv_Classifier.predict(X_test)


# In[ ]:


print(Sv_Classifier.score(X_test, y_test))
print(Sv_Classifier.score(X_combined_sampling, y_combined_sampling))


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,Sv_predictions)


# # Naive Bayes

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()


# In[ ]:


nb=gnb.fit(X_combined_sampling, y_combined_sampling)
nb


# In[ ]:


y_pred = gnb.predict(X_test)
y_pred


# In[ ]:


print(gnb.score(X_test, y_test))
print(gnb.score(X_combined_sampling, y_combined_sampling))


# In[ ]:


#overfitting


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# # Conclusion

# Random forest and naive bayes models are overfitting due to imbalanced data. It can be reduced by sampling techniques.Also,we can randomized search cv to find the best parametre

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('knn', model3)], voting='hard')


# In[ ]:


print('\n                     Accuracy     Error')
print('                     ----------   --------')
print('Logistic Regression    : {:.04}%       {:.04}%'.format( model.score(X_test, y_test)* 100,                                                  100-(model.score(X_test, y_test) * 100)))

print('KNN                    : {:.04}%       {:.04}% '.format(knn_classifier.score(X_test, y_test) * 100,                                                        100-(knn_classifier.score(X_test, y_test) * 100)))

print('Random Forest          : {:.04}%       {:.04}% '.format(rf_classifier.score(X_test, y_test)* 100,                                                           100-(rf_classifier.score(X_test, y_test)* 100)))
print('Naivebayes             : {:.04}%      {:.04}% '.format(gnb.score(X_test, y_test)* 100,                                                           100-(gnb.score(X_test, y_test)* 100)))
print('Support Vector Machine : {:.04}%     {:.04}% '.format(Sv_Classifier.score(X_test, y_test)* 100,                                                           100-(Sv_Classifier.score(X_test, y_test)* 100)))

