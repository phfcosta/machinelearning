#!/usr/bin/python

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from classify import classify, accuracy


import numpy as np
import pylab as pl

features_train, label_train, features_test, labels_test = makeTerrainData()

grade_fast = [features_train[ii][0] for ii in range(0,len(features_train)) if label_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0,len(features_train)) if label_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0,len(features_train)) if label_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0,len(features_train)) if label_train[ii]==1]

clf = classify(features_train, label_train)
clf.fit(features_train,label_train)
pred = clf.predict(features_train)

acuracia = accuracy(pred,label_train)
print("acuracia" + str(acuracia))

prettyPicture(clf,features_test,labels_test)
output_image("test.png","png",open("test.png","rb").read())
