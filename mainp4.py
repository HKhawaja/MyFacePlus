from numpy import loadtxt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    #tr_file = open("posts_train.txt", "r")
    #tr = tr_file.read().split(',')
    #print(tr)

    tr_init = loadtxt("posts_train.txt", comments="#", delimiter=",", unpack=False)

    # there are 7 variables

    y_tr = np.empty((len(tr_init), 2))
    temp = tr_init[:, 5]
    y_tr[:, 0] = temp
    temp2 = tr_init[:, 6]
    y_tr[:, 1] = temp2

    X_tr = np.empty((len(tr_init), 5))
    for a in range(5):
        temp = tr_init[:, a]
        X_tr[:, a] = temp

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    X_te = np.empty((len(te_init), 5))
    for a in range(5):
        temp = te_init[:, a]
        X_te[:, a] = temp

    print(len(X_tr), len(X_te))