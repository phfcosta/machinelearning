#!/usr/bin/python

import numpy
from class_vis import prettyPicture, output_image
import matplotlib.pyplot as plt

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData() 


from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

km_net_worth = reg.predict([27][0])[0][0]

slope = reg.coef_[0][0]

intercept = reg.intercept_[0]

test_score = reg.score(ages_test, net_worths_test)

training_score = reg.score(ages_train, net_worths_train)

def submitFit():
	return {"networth":km_net_worth,
				"slope":slope,
				"intercept":intercept,
				"stats on test":test_score,
				"stats on training":training_score}

print(submitFit())