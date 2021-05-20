
import numpy as np

import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB as BE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
X, Y = fetch_openml('mnist_784', version=1, data_home='./scikit_learn_data', return_X_y=True)
# make the value of pixels from [0, 255] to [0, 1] for further process
X = X / 255.
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
be=BE()

be.fit(X_train,Y_train)

test_accuracy=be.score(X_test,Y_test)
train_accuracy=be.score(X_train,Y_train)
print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))