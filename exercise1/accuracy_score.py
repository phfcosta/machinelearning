from sklearn.metrics import accuracy_score

def accuracy(label_train,pred):
	return accuracy_score(label_train,pred)