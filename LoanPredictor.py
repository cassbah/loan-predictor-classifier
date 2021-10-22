import numpy as np 
import csv 
import pandas as pd
import json as j
import re
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import xgboost
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import base64


pd.set_option("display.max_columns", None)

FILE_NAME2="traindemographics.csv"
FILE_NAME3="trainprevloans.csv"
FILE_NAME1="trainperf.csv"

#reading data from csv files
data1=pd.read_csv(FILE_NAME1,delimiter=",",na_values=[""])
data2=pd.read_csv(FILE_NAME2,delimiter=",",na_values=[""])
data3=pd.read_csv(FILE_NAME3,delimiter=",",na_values=[""])

#merging the data sets into one by matching on customer id
combined_data=pd.merge(data1,data2,left_on="customerid",right_on="customerid",how="left")
#combined_data=pd.merge(combined_data_try,data3,left_on="customerid",right_on="customerid",how="inner")

#print(combined_data2.isnull().sum())

#data cleanup and preprocessing
#filling all null values with averages.
combined_data['referredby'] = combined_data["referredby"].fillna(combined_data['referredby'].mode()[0])
combined_data['bank_branch_clients'] = combined_data["bank_branch_clients"].fillna(combined_data['bank_branch_clients'].mode()[0])
combined_data['employment_status_clients'] = combined_data["employment_status_clients"].fillna(combined_data['employment_status_clients'].mode()[0])
combined_data['level_of_education_clients'] = combined_data["level_of_education_clients"].fillna(combined_data['level_of_education_clients'].mode()[0])
#combined_data['referredby_y'] = combined_data["referredby_y"].fillna(combined_data['referredby_y'].mode()[0])
#print(combined_data.shape)



#checking where null values are and how many per column
#print(combined_data.isnull().sum())


#some data visualisations
#sns.countplot(combined_data['level_of_education_clients'])
#plt.show()

#correlation matrix

#corr = combined_data.corr()
#plt.figure(figsize=(15,10))
#sns.heatmap(corr, annot = True, cmap="BuPu")
#plt.show()

#encoding data types for columns which are labelled as objects
cols = ["good_bad_flag","customerid","referredby",'birthdate',"bank_account_type","bank_name_clients","bank_branch_clients","employment_status_clients","level_of_education_clients","approveddate","creationdate"]#,"closeddate","firstduedate","firstrepaiddate"]
le = LabelEncoder()
for col in cols:
    combined_data[col] = le.fit_transform(combined_data[col])

#print(combined_data.head())

corr = combined_data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot = True, cmap="BuPu")
plt.show()
#most important features obtained from correlation matrix
Features=['loanamount','totaldue','referredby','loannumber','employment_status_clients']

#getting training and test data and splitting it 80/20
X=combined_data[Features]
y=combined_data['good_bad_flag']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=2)

#using gradient boosting classifier with training and test data from above
gradientboost_clf = GradientBoostingClassifier(max_depth=1,max_features=4,learning_rate=0.001, random_state=1)
gradientboost_clf.fit(X_train,y_train)
gbpred = gradientboost_clf.predict(X_test)
gbacc = accuracy_score(y_test, gbpred)

print("accuracy score of the gradient boosting clf  is :")
print(gbacc)
print(classification_report(y_test,gbpred))
print(confusion_matrix(y_test,gbpred))

#Neural network
nn = MLPClassifier(activation="logistic",solver="sgd",hidden_layer_sizes=(200,300),random_state=1, max_iter=400,)
nn.fit(X_train,y_train)
nnpred=nn.predict(X_test)
nnacc=accuracy_score(y_test,nnpred)

print("accuracy score of the neural_network clf  is :")
print(nnacc)
print(classification_report(y_test,nnpred))
print(confusion_matrix(y_test,nnpred))



#Testing
FILE_NAME2T="testdemographics.csv"
FILE_NAME3T="testprevloans.csv"
FILE_NAME1T="testperf.csv"
data1T=pd.read_csv(FILE_NAME1T,delimiter=",",na_values=[""])
data2T=pd.read_csv(FILE_NAME2T,delimiter=",",na_values=[""])
data3T=pd.read_csv(FILE_NAME3T,delimiter=",",na_values=[""])

newx=data1T['customerid']
#merging the data sets into one by matching on customer id
combined_dataT=pd.merge(data1T,data2T,left_on="customerid",right_on="customerid",how="left")

#data cleanup and preprocessing
#filling all null values with averages.
combined_dataT['referredby'] = combined_dataT["referredby"].fillna(combined_dataT['referredby'].mode()[0])
combined_dataT['bank_branch_clients'] = combined_dataT["bank_branch_clients"].fillna(combined_dataT['bank_branch_clients'].mode()[0])
combined_dataT['employment_status_clients'] = combined_dataT["employment_status_clients"].fillna(combined_dataT['employment_status_clients'].mode()[0])
combined_dataT['level_of_education_clients'] = combined_dataT["level_of_education_clients"].fillna(combined_dataT['level_of_education_clients'].mode()[0])

#encoding data types for columns which are labelled as objects
cols = ["referredby",'birthdate',"bank_account_type","bank_name_clients","bank_branch_clients","employment_status_clients","level_of_education_clients","approveddate","creationdate"]#,"closeddate","firstduedate","firstrepaiddate"]
le = LabelEncoder()
for col in cols:
    combined_dataT[col] = le.fit_transform(combined_dataT[col])


#most important features
Features=['loanamount','totaldue','referredby','loannumber','employment_status_clients']

X=combined_dataT[Features]
mypred=gradientboost_clf.predict(X)
mynewdata={'customerid': newx, 'good_bad_flag':mypred}
mynewdataT=pd.DataFrame(mynewdata)
mynewdataT=mynewdataT.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
#creating  a new csv file with test results
#mynewdataT.to_csv('nn3finalsubmission.csv',index=False)



