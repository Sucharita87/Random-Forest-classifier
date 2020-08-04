import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

data= pd.read_csv("F:\\ExcelR\\Assignment\\Decision Tree\\Company_Data.csv")
data.describe() # need to scale the data
data.info() #float64(1), int64(7), object(3)
data.isnull().sum()
categorical = [var for var in data.columns if data[var].dtype =='O']
print('There are {} categorical variables.\n'.format(len(categorical)))
print('The categoricall variables are:\n\n',categorical)
string_values =  ['ShelveLoc', 'Urban', 'US']
data[categorical].isnull().sum()
data.head()
plt.hist(data['Sales']) # max sales are in range of 6-10 k; three sales group are there (0-6)k, (>6-10)k, (>0)k
# as we have to decide the influencing factors for good sales, we have to classify sales

stype = []  # define a new value
for value in data["Sales"]: 
    if value <6: 
        stype.append("bad") 
    else: 
        stype.append("good") 
       
data["stype"] = stype    # add stype to the  dataframe of "data" 
plt.hist(data['stype'])

# label encode all categorical variables

string_values =  ['ShelveLoc', 'Urban', 'US','stype']
n = preprocessing.LabelEncoder()
for i in string_values:
    data[i]= n.fit_transform(data[i])

data.columns
data.drop(["Sales"], axis =1, inplace = True)
data.columns
c= data.columns
data.shape # 400, 11
# train test split
train,test = train_test_split(data, test_size =0.2)
x_train = train.iloc[:,0:10]
y_train = train.iloc[:,10:11]
x_test = test.iloc[:,0:10]
y_test = test.iloc[:,10:11]
y_train.stype.value_counts() # 1:215 and 0: 105
y_test.stype.value_counts() # 1:55 and 0: 25

# data set is not totally  imbalanced

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=3,oob_score=True,n_estimators=100,criterion="entropy")

#### Attributes that comes along with RandomForest function
rf.fit(x_train, y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # array
rf.n_classes_ # 2 levels
rf.n_features_  # Number of input features in model is 10

rf.n_outputs_ # Number of outputs when fit performed is 1

rf.oob_score_  # 82.5%
rf.predict(x_train)
len(rf.predict(x_train))
type(rf.predict(x_train))

pred =rf.predict(x_train)
y_pred = pd.DataFrame(pred) # convert to dataframe so that it can be added in 
type(y_pred)
type(x_train_array)
data['rf_pred'] = y_pred # add predicted value to main file
confusion_matrix(y_train,y_pred) # Confusion matrix
#[[105,   0],
#       [  0, 215]]
print(classification_report(y_train, y_pred)) # accuracy 100%

y_test_pred = rf.predict(x_test)
confusion_matrix(y_test,y_test_pred)
#[[12, 13],
# [ 5, 50]]
print(classification_report(y_test,y_test_pred)) # accuracy 78%
