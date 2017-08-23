#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print 'num of features', len(features_train[0])  # 3785


#########################################################
### your code goes here ###

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)

t0 = time()

# speed up training
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t1, 3), "s"

print 'pred', pred[10], pred[26], pred[50]

count = 0
for p in pred:
    if p == 1:
        count += 1
print 'count', count  # 877


from sklearn.metrics import accuracy_score
accu = accuracy_score(pred, labels_test)
print 'accu', accu


#########################################################


"""
3875 features

training time: 76.532 s
prediction time: 0.074 s
pred 1 0 1
count 870
accu 0.977815699659
"""

"""
379 features

training time: 5.338 s
prediction time: 0.002 s
pred 1 0 1
count 887
accu 0.967007963595
"""