#!/usr/bin/python
import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = classify(features_train, labels_train,2)
pred = clf.predict(features_test)

clf50 = classify(features_train, labels_train, 50)
pred50 = clf50.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test,pred)
acc50 = accuracy_score(labels_test,pred50)


def submitAccuracy():
	return {"acc":round(acc,3)," acc50": round(acc50,3)}

print(submitAccuracy())

prettyPicture(clf, features_test,labels_test)