import sys
from class_vis import pretty_picture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

features_train, label_train, features_test, labels_test = makeTerrainData()

from sklearn.svm import SVC
clf = SVC(kernel="linear")



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,labels_test)

def submitAccuracy():
	return acc