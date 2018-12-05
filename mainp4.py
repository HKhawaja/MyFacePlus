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
    temp = tr_init[:, 4]
    y_tr[:, 0] = temp
    temp2 = tr_init[:, 5]
    y_tr[:, 1] = temp2

    X_tr = np.empty((len(tr_init), 4)) # Does not take id as a variable
    for a in range(3): # this weird range plus the end is because of how the dataset was set up
        temp = tr_init[:, a+1]
        X_tr[:, a] = temp
    X_tr[:, 3] = tr_init[:, 6]

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    X_te = np.empty((len(te_init), 4))
    for a in range(4):
        temp = te_init[:, a+1]
        X_te[:, a] = temp

    # Time for some data cleaning
    # Remove users on "Null island (1057 users)
    counts = []
    for k in range(len(y_tr)):
        if y_tr[k][0] == 0 and y_tr[k][1] == 0:
            counts.append(k)
    for a in range(len(counts)-1, -1, -1): # count backwards
        k = counts[a]
        y_tr = np.delete(y_tr, k, 0)
        X_tr = np.delete(X_tr, k, 0)
    #Now there are no elements left on null island