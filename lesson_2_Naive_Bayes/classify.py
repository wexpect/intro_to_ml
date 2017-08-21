from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def classify(features_train, labels_train):
	### import the sklearn module for GaussianNB
	### create classifier
	### fit the classifier on the training features and labels
	### return the fit classifier

	### your code goes here!

	# Naive Bayes
	# clf = GaussianNB()	# 0.884

	# SVM
	# clf = SVC(kernel='linear')	# 0.92
	# clf = SVC(C=.03, kernel='linear')	# 0.852
	# clf = SVC(C=10, kernel='linear')	# 0.916

	clf = SVC(kernel='rbf')	 # 0.92
	# clf = SVC(C=0.03, kernel='rbf')
	# clf = SVC(C=10, kernel='rbf')

	"""
	gamma defines how far the influence of a single example reaches
	low value: far reach
	high value: close reach. only consider points near decision boundary	
	"""
	# big gamma gives more intricate decision boundaries
	# clf = SVC(kernel='rbf', gamma=500)	# 0.944
	# clf = SVC(kernel='rbf', gamma=0.01)	# 716

	clf.fit(features_train, labels_train)
	return clf
