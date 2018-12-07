from numpy import loadtxt
import numpy as np
from learners import *
import sklearn
from sklearn.linear_model import LinearRegression


def clean_up_graph(ids_lat_lng):
    graph = loadtxt("graph.txt", comments='#', delimiter="\t", unpack=False)

    # remove from graph any line on which either or both ids is/are not found in our original list of posts
    print('cleaning up graph')
    for i in range(len(graph)-1, -1, -1):
        if graph[i][0] not in ids_lat_lng or graph[i][1] not in ids_lat_lng:
            graph = np.delete(graph, i, 0)
    print(graph)

    # friend_dict: key = id and value = list of friend ids
    friend_dict = {}
    for i in range(len(graph) -1, -1, -1):
        source = graph[i][0]
        dest = graph[i][1]
        if source not in friend_dict:
            dests = [dest]
            friend_dict[source] = dests
        else:
            dests = friend_dict[source]
            dests.append(dest)
            friend_dict[source] = dests

    # find median latitude and longitude of each id's friends
    # median_lat_lng_friends: dict where key = id and value = tuple of median of latitude and longitude of friends
    median_lat_lng_friends = {}
    for id in ids_lat_lng:
        friends = friend_dict[id]
        lats = []
        longs = []
        for friend in friends:
            lat, lng = ids_lat_lng[friend]
            lats.append(lat)
            longs.append(lng)
        median_lat_lng_friends[id] = (np.median(lats), np.median(longs))


def posts_cleanup():
    tr_init = loadtxt("posts_train.txt", comments="#", delimiter=",", unpack=False) # Full training set from txt file

    # there are 7 variables
    # y_tr: matrix with cols: latitude, longitude and row for each id
    y_tr = np.empty((len(tr_init), 2)) # Create empty array which will hold the response variable
    temp = tr_init[:, 4]
    y_tr[:, 0] = temp
    temp2 = tr_init[:, 5]
    y_tr[:, 1] = temp2
    # The above 4 lines take the correct values (latitude and longitude) and add them to the response np array

    # X_tr: (Hour1, Hour2, Hour3, numPosts)
    X_tr = np.empty((len(tr_init), 4)) # Does not take id as a variable
    X_tr_ids = np.array(tr_init[:, 0]).astype(int) # This is a list of ids that we may need later
    for a in range(3): # this weird range plus the end is because of how the dataset was set up
        temp = tr_init[:, a+1]
        X_tr[:, a] = temp
    X_tr[:, 3] = tr_init[:, 6]

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    # Then, we do the same thing with the test data; start at column 1 and get a 4-column dataset with
    # the columns as hour1, Hour2, Hour3, numposts
    X_te = np.empty((len(te_init), 4))
    X_te_ids = np.array(te_init[:, 0]).astype(int) # This is a list of ids to use later
    for a in range(4):
        temp = te_init[:, a+1]
        X_te[:, a] = temp

    # Remove users on "Null island" (1057 users) from X_tr, X_tr_ids and y_tr
    # X_tr, X_tr_ids and y_tr have same length
    counts = []
    for k in range(len(y_tr)):
        if y_tr[k][0] == 0 and y_tr[k][1] == 0:
            counts.append(k)

    for a in range(len(counts)-1, -1, -1): # count backwards
        k = counts[a]
        y_tr = np.delete(y_tr, k, 0)
        X_tr = np.delete(X_tr, k, 0)
        X_tr_ids = np.delete(X_tr_ids, k, 0)

    # ids_lat_lng: key: id and value: (latitude, longitude) for training set
    ids_lat_lng = {}
    for i in range(len(y_tr)):
        ids_lat_lng[X_tr_ids[i]] = (y_tr[i][0], y_tr[i][1])

    # return ids_lat_lng
    #return [X_tr, y_tr, X_te, X_te_ids]
    return ids_lat_lng

"""
have you seen this? https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression 
"""
def linearRegressor(tr_X, tr_y, te_X):  # Code to play around with for performing a linear regression
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linearRegression.predict(te_X)
    return preds

# The above method uses a simple linear model, which we can edit or make new methods to make more complex,
# and outputs the linear model's prediction

if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    #tr_file = open("posts_train.txt", "r")
    #tr = tr_file.read().split(',')
    #print(tr)
    clean_up_graph(posts_cleanup())
