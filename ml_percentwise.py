#!/usr/bin/python3

from sklearn.datasets import load_iris

#loading iris data set

iris =load_iris()

#now splitting into test and train data sets

from sklearn.model_selection import train_test_split

x,y,z,a=train_test_split(iris.data,iris.target,test_size=0.1)


'''
vcvcxvxvxvx
'''

#calling decision tree classifier

from sklearn import tree

dsclf = tree.DecisionTreeClassifier()

dsclf = tree.DecisionTreeClassifier()

#now training data with decision

trained = dsclf.fit(x,z)

#now time for prediction

output = trained.predict(y)
print(output)

#checking % of accuracy 
from sklearn.metrics import accuracy_score

check_pct = accuracy_score(a,output)
print(check_pct)

