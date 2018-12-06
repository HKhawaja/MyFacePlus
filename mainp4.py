from numpy import loadtxt
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# SVC is for Support Vector Classifier -- we called it SVM in class
from sklearn.tree import DecisionTreeClassifier

def clean_up_graph(ids):
    graph = loadtxt("graph.txt", comments='#', delimiter="\t", unpack=False)
    """
    want to remove from graph:
    1) any entries where either id is an id not found in our original list of posts
    """
    print('cleaning up graph')
    for i in range(len(graph)-1, -1, -1):
        if graph[i][0] not in ids or graph[i][1] not in ids:
            graph = np.delete(graph, i, 0)
    print(graph)

    # create list of friend ids
    friend_list = {}
    for i in range(len(graph) -1, -1, -1):
        source = graph[i][0]
        dest = graph[i][1]
        if source not in friend_list:
            dests = [dest]
            friend_list[source] = dests
        else:
            dests = friend_list[source]
            dests.append(dest)
            friend_list[source] = dests

    # find median latitude and longitude of each id's 'friends'
    median_lat_lng = {}


def posts_cleanup():
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

    """
    we would need to get the user id too right, in order to correlate that user
    to his social graph?
    """
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

    ids_lat_lng = np.empty((len(y_tr), 3))
    ids_lat_lng[0] =
    return set(tr_init[:,0])


if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    #tr_file = open("posts_train.txt", "r")
    #tr = tr_file.read().split(',')
    #print(tr)
    clean_up_graph(posts_cleanup())