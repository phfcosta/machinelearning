from sklearn import tree

def classify(features_train, labels_train,split):

	clf = tree.DecisionTreeClassifier(min_samples_split=split)
	clf.fit(features_train,labels_train)

	return clf