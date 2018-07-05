from sklearn.naive_bayes import GaussianNB

def classify(feature_train, label_train):
	clf = GaussianNB()
	clf.fit(feature_train,label_train)
	return clf
