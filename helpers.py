from numpy import loadtxt
import numpy as np
import sklearn


def clean_up_graph(test_set, ids_lat_lng):
    graph = loadtxt("graph.txt", comments='#', delimiter="\t", unpack=False)
    graph_te = loadtxt("graph.txt", comments='#', delimiter="\t", unpack=False)
    X_te_ids = test_set[3]
    # Print statements show that these are indeed the same

    # remove from graph any line on which either or both ids is/are not found in our original list of posts
    # for test ids, remove any line which does not have first entry as test id and second as training id
    for i in range(len(graph)-1, -1, -1):
        if graph[i][0] not in ids_lat_lng or graph[i][1] not in ids_lat_lng:
            graph = np.delete(graph, i, 0)

    for i in range(len(graph_te)-1, -1, -1):
        if graph_te[i][0] not in X_te_ids or graph_te[i][1] not in ids_lat_lng:
            graph_te = np.delete(graph_te, i, 0)
    # 356268 elements in graph, 6523 in teh test graph
    # And all the  elements in the test graph second column are in the training set as they should be

    # friend_dict: key = id and value = list of friend ids
    friend_dict = {}
    friend_dict_te = {}
    for i in range(len(graph) - 1, -1, -1):
        source = graph[i][0]
        dest = graph[i][1]
        if source not in friend_dict:
            dests = [dest]
            friend_dict[source] = dests
        else:
            dests = friend_dict[source]
            dests.append(dest)
            friend_dict[source] = dests

    # create friend list for test ids
    for i in range(len(graph_te) -1, -1,-1):
        source_te = graph_te[i][0]
        dest_te = graph_te[i][1]
        if source_te not in friend_dict_te:
            dests = [dest_te]
            friend_dict_te[source] = dests
        else:
            dests = friend_dict_te[source]
            dests.append(dest_te)
            friend_dict_te[source] = dests


    # if id is present in original set but not in graph, it has 0 friends
    for id in ids_lat_lng:
        if id not in friend_dict:
            friend_dict[id] = []

    # likewise for test ids
    for id in X_te_ids:
        if id not in friend_dict_te:
            friend_dict_te[id] = []

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

    # likewise for test ids
    # key insight is that only want friends of test id that were present in training set
    median_lat_lng_friends_te = {}
    for id in X_te_ids:
        friends = friend_dict_te[id]
        lats_te = []
        longs_te = []
        for friend in friends:
            lat, lng = ids_lat_lng[friend]
            lats_te.append(lat)
            longs_te.append(lng)
        median_lat_lng_friends_te[id] = (np.median(lats_te), np.median(longs_te))

    X_tr_lat_lng = np.empty((len(ids_lat_lng),3))
    X_te_lat_lng = np.empty((len(X_te_ids),3))
    i = 0
    for id in ids_lat_lng:
        lat, lng = median_lat_lng_friends[id]
        X_tr_lat_lng[i][0] = id
        X_tr_lat_lng[i][1] = lat
        X_tr_lat_lng[i][2] = lng
        i = i+1

    i = 0
    for id in X_te_ids:
        lat_te, lng_te = median_lat_lng_friends_te[id]
        X_te_lat_lng[i][0] = id
        X_te_lat_lng[i][1] = lat_te
        X_te_lat_lng[i][2] = lng_te
        i = i + 1

    np.savetxt('ids_lat_lng_tr.txt', X_tr_lat_lng, delimiter=',')
    np.savetxt('ids_lat_lng_te.txt', X_te_lat_lng, delimiter=',')

def remove_null_island(tr_X, tr_y, tr_X_ids):

    # Remove users on "Null island" (1057 users) from X_tr, X_tr_ids and y_tr
    # X_tr, X_tr_ids and y_tr have same length
    counts = []
    for k in range(len(tr_y)):
        if tr_y[k][0] == 0 and tr_y[k][1] == 0:
            counts.append(k)

    for a in range(len(counts)-1, -1, -1): # count backwards
        k = counts[a]
        tr_y = np.delete(tr_y, k, 0)
        tr_X = np.delete(tr_X, k, 0)
        tr_X_ids = np.delete(tr_X_ids, k, 0)
    return tr_X, tr_y, tr_X_ids

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
    X_tr_ids = np.array(tr_init[:, 0]).astype(int) # This is a list of ids that we may need later
    X_tr = np.empty((len(tr_init), 4))
    for a in range(3): # this weird range plus the end is because of how the dataset was set up
        temp = tr_init[:, a+1]
        X_tr[:, a] = temp

    X_tr[:, 3] = tr_init[:, 6]

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    # Then, we do the same thing with the test data; start at column 1 and get a 4-column dataset with
    # the columns as hour1, Hour2, Hour3, numposts
    X_te_ids = np.array(te_init[:, 0]).astype(int) # This is a list of ids to use later
    X_te = np.empty((len(te_init), 4))
    for a in range(4):
        temp = te_init[:, a+1]
        X_te[:, a] = temp

    X_tr, y_tr, X_tr_ids = remove_null_island(X_tr, y_tr, X_tr_ids)

    # ids_lat_lng: key: id and value: (latitude, longitude) for training set
    ids_lat_lng = {}
    for i in range(len(y_tr)):
        ids_lat_lng[X_tr_ids[i]] = (y_tr[i][0], y_tr[i][1])

    return [X_tr, y_tr, X_te, X_te_ids], ids_lat_lng

def posts_cleanup_medians():
    # This method outputs a training and test set including the median latitude and longitude of each user's friends
    tr_init = loadtxt("posts_train.txt", comments="#", delimiter=",", unpack=False) # Full training set from txt file
    tr_lat_lng_friends = loadtxt("ids_lat_lng_tr.txt", comments="#", delimiter=",",  unpack=False)
    te_lat_lng_friends = loadtxt("ids_lat_lng_te.txt", comments='#', delimiter=",", unpack=False)
    # there are 7 variables
    # y_tr: matrix with cols: latitude, longitude and row for each id
    y_tr = np.empty((len(tr_init), 2)) # Create empty array which will hold the response variable
    temp = tr_init[:, 4]
    y_tr[:, 0] = temp
    temp2 = tr_init[:, 5]
    y_tr[:, 1] = temp2
    # The above 4 lines take the correct values (latitude and longitude) and add them to the response np array

    # X_tr: (Hour1, Hour2, Hour3, numPosts)
    X_tr_ids = np.array(tr_init[:, 0]).astype(int) # This is a list of ids that we may need later
    X_tr = np.empty((len(tr_init), 6))
    for a in range(3): # this weird range plus the end is because of how the dataset was set up
        temp = tr_init[:, a+1]
        X_tr[:, a] = temp

    X_tr[:, 3] = tr_init[:, 6]
    X_tr, y_tr, X_tr_ids = remove_null_island(X_tr, y_tr, X_tr_ids)
    X_tr[:, 4] = tr_lat_lng_friends[:, 1]
    X_tr[:, 5] = tr_lat_lng_friends[:, 2] #these lines add the median lat, long of friends as variables

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    # Then, we do the same thing with the test data; start at column 1 and get a 6-column dataset with
    # the columns as hour1, Hour2, Hour3, numposts, med_lat, med_long
    X_te_ids = np.array(te_init[:, 0]).astype(int) # This is a list of ids to use later
    X_te = np.empty((len(te_init), 6))
    for a in range(4):
        temp = te_init[:, a+1]
        X_te[:, a] = temp
    X_te[:, 4] = te_lat_lng_friends[:, 1] #this should be all the test users and their friends' median lat and long
    X_te[:, 5] = te_lat_lng_friends[:, 2]
    # ids_lat_lng: key: id and value: (latitude, longitude) for training set
    ids_lat_lng = {}
    for i in range(len(y_tr)):
        ids_lat_lng[X_tr_ids[i]] = (y_tr[i][0], y_tr[i][1])

    return [X_tr, y_tr, X_te, X_te_ids], ids_lat_lng

"""
have you seen this? https://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression 
"""

# The above method uses a simple linear model, which we can edit or make new methods to make more complex,
# and outputs the linear model's prediction

if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    #tr_file = open("posts_train.txt", "r")
    #tr = tr_file.read().split(',')
    #print(tr)
    a, b = posts_cleanup()
    clean_up_graph(a, b)
