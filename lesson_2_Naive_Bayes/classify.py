from time import time


def classify(features_train, labels_train):
	### import the sklearn module for GaussianNB
	### create classifier
	### fit the classifier on the training features and labels
	### return the fit classifier

	### your code goes here!

	# ------- Naive Bayes --------
	# from sklearn.naive_bayes import GaussianNB
	# clf = GaussianNB()	# 0.884


	# ------- SVM --------
	# from sklearn.svm import SVC
	# clf = SVC(kernel='linear')	# 0.92
	# clf = SVC(C=.03, kernel='linear')	# 0.852
	# clf = SVC(C=10, kernel='linear')	# 0.916

	"""
	C: controls tradeoff between smooth decision boundary
	and classifying training points correctly
	small C: smooth.
	large C: more training points correct
	"""
	# clf = SVC(kernel='rbf')	 # 0.92
	# clf = SVC(C=0.03, kernel='rbf')  # 0.792
	# clf = SVC(C=10, kernel='rbf')	# 0.912


	"""
	Gamma: defines how far the influence of a single example reaches
	small Gamma: far reach. Smooth.
	large Gamma: close reach. Only consider points near decision boundary. Ends
	up with more intricate decision boundary
	"""
	# clf = SVC(kernel='rbf', gamma=0.01)	# 0.716
	# clf = SVC(kernel='rbf', gamma=500)	# 0.944



	# ------- K Nearest Neighbors --------
	# from sklearn.neighbors import KNeighborsClassifier
	# clf = KNeighborsClassifier()  # 0.92



	# ------- Decision Tree --------
	# from sklearn.tree import DecisionTreeClassifier
	# # clf = DecisionTreeClassifier()  # 0.908
	# clf = DecisionTreeClassifier(min_samples_split=50)  # 0.912



	# ------- Random Forest --------
	from sklearn.ensemble import RandomForestClassifier
	# clf = RandomForestClassifier()  # 0.924
	# clf = RandomForestClassifier(
	# 	n_estimators=100,
	# 	# max_depth=None,
	# 	# min_samples_split=50,
	# )  # 0.92


	# ------- Gradient Boosting Decision Trees --------
	# from sklearn.ensemble import GradientBoostingClassifier
	# # clf = GradientBoostingClassifier()  # 0.912
	# clf = GradientBoostingClassifier(
	# 	n_estimators=100,
	# 	# min_samples_split=50,
	# 	max_depth=10,
	# )  # 0.92



	# ------- model fit --------
	t0 = time()
	clf.fit(features_train, labels_train)
	print "training time:", round(time() - t0, 3), "s"

	# print 'feature_importances', clf.feature_importances_

	return clf

# 93.6