#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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


#########################################################
### your code goes here ###

#########################################################

from sklearn.svm import SVC

# clf = SVC(kernel='linear')  # accu 0.984072810011

# clf = SVC(kernel='rbf')
"""
training time: 0.125 s
prediction time: 1.305 s
accu 0.616040955631
"""

# clf = SVC(C=10, kernel='rbf')
"""
training time: 0.115 s
prediction time: 1.271 s
accu 0.616040955631
"""

# clf = SVC(C=100, kernel='rbf')
"""
training time: 0.123 s
prediction time: 1.374 s
accu 0.616040955631
"""

# clf = SVC(C=1000, kernel='rbf')
"""
training time: 0.111 s
prediction time: 1.183 s
accu 0.821387940842
"""

clf = SVC(C=10000, kernel='rbf')
"""
training time: 0.111 s
prediction time: 1.06 s
accu 0.892491467577
"""

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

"""
linear

training time: 219.762 s
prediction time: 19.292 s
accu 0.984072810011
"""

"""
linear

1/100 training data

training time: 0.127 s
prediction time: 1.111 s
accu 0.884527872582
"""

"""
rbf

training time: 122.542 s
prediction time: 11.69 s
accu 0.990898748578
"""

"""
rbf

1/100 training data

training time: 0.125 s
prediction time: 1.305 s
accu 0.616040955631
"""