from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def classify(feature_train, label_train):
	clf = GaussianNB()
	clf.fit(feature_train,label_train)
	return clf

def accuracy(label_train,pred):
	return accuracy_score(pred,label_train)