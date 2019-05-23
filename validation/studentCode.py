#!/usr/bin/python

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
features = iris.data
labels = iris.target

from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features,labels,test_size=0.4,random_state=0)

clf = SVC(kernel="linear",C=1.)
clf.fit(features_train,labels_train)

print (clf.score(features_test,labels_test))
