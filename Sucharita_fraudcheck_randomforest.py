import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
fraud= pd.read_csv("F:\\ExcelR\\Assignment\\Decision Tree\\Fraud_check.csv")
fraud.info() # 3 object, 3 int
fraud.describe()
fraud.shape
fraud.isnull().sum()

categorical = [var for var in fraud.columns if fraud[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
string_values = ['Undergrad', 'Marital.Status', 'Urban']
fraud[categorical].isnull().sum() # shows no null values
fraud.rename(columns= {"Taxable.Income":"y"}, inplace = True)
plt.hist(fraud['y'])  


# add income column which will serve as the classifier
fraud["income"]="<=30000"
fraud.loc[fraud["y"]>=30000,"income"]="Good"
fraud.loc[fraud["y"]<=30000,"income"]="Risky"
plt.hist(fraud['income'])
fraud.income.value_counts() # good: 476, risky:124
fraud.income.value_counts().plot(kind="pie") 

# label encoding of all string values(also include "income")
string_new= ['Undergrad', 'Marital.Status', 'Urban','income']
number= preprocessing.LabelEncoder()
for i in string_new:
   fraud[i] = number.fit_transform(fraud[i])
   

# now that classification is done as "good" and "Risky", we will drop the taxable income(y) column
fraud.drop(["y"], axis =1, inplace = True)
fraud.describe()
train,test = train_test_split(fraud, test_size = 0.2)
x_train = train.iloc[:,[0,2,3]] # initially there was overfit problem, so removed few variables
y_train = train.iloc[:,5:6]
x_test = test.iloc[:,[0,2,3]]
y_test = test.iloc[:,5:6]
y_train.income.value_counts()# 0:384   1:96
y_test.income.value_counts() # 0:99   1:21

#imbalance dataset handling
sm = SMOTE(random_state = 2) 
x_train1, y_train1 = sm.fit_sample(x_train, y_train)  
print('After OverSampling, the shape of x_train: {}'.format(x_train1.shape)) #(768,5)
print('After OverSampling, the shape of y_train: {} \n'.format(y_train1.shape)) #(768, 1)
y_train1.income.value_counts() # 0:384   1:384

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")

#### Attributes that comes along with RandomForest function
rf.fit(x_train, y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # array
rf.n_classes_ # 2 levels
rf.n_features_  # Number of input features in model is 5

rf.n_outputs_ # Number of outputs when fit performed is 1

rf.oob_score_  # 76.45%
rf.predict(x_train)
len(rf.predict(x_train))
type(rf.predict(x_train))

pred =rf.predict(x_train)
y_pred = pd.DataFrame(pred) # convert to dataframe so that it can be added in 
type(y_pred)
fraud['rf_pred'] = y_pred # add predicted value to main file
confusion_matrix(y_train,y_pred) # Confusion matrix
#[[384,   0],
# [  0,  96]]
print(classification_report(y_train, y_pred)) # accuracy 100%
y_test_pred = rf.predict(x_test)
confusion_matrix(y_test,y_test_pred)
#[[86,  9],
# [23,  2]]
print(classification_report(y_test,y_test_pred)) # accuracy 73%


