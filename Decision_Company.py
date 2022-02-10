import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

company = pd.read_csv("file:///D:/DATA SCIENCE/ExcelR/Assignments/Decision Tree/Company_Data.csv")
company.head()

company['Sales'].unique()
company.Sales.value_counts()
plt.hist(company.Sales)
# here we need to change the sales cloumn to Discrete data
# 
# using loc command
company.loc[company['Sales'] < 10, 'Highsales'] = 0
company.loc[company['Sales'] > 10, 'Highsales'] = 1

company.loc[company['Highsales'] == 0, 'Highsales'] = "No"
company.loc[company['Highsales'] == 1, 'Highsales'] = "Yes"

company.drop(["Sales"], axis = 1, inplace = True)
colnames = list(company.columns)

predictors = colnames[0:10]

target = colnames[10]

# Splitting data into training and testing data set

# np.random.uniform(start,stop,size) will generate array of real numbers with size = size
company['is_train'] = np.random.uniform(0, 1, len(company))<= 0.75
company['is_train']
train,test = company[company['is_train'] == True],company[company['is_train']==False]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company,test_size = 0.2)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

# Accuracy = train
np.mean(train.Species == model.predict(train[predictors]))

# Accuracy = test

np.mean(test.Species == model.predict(test[target]))











