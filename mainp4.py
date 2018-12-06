from numpy import loadtxt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

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
    tr_init = loadtxt("posts_train.txt", comments="#", delimiter=",", unpack=False) # Full training set from txt file

    # there are 7 variables

    y_tr = np.empty((len(tr_init), 2)) # Create empty array which will hold the response variable
    temp = tr_init[:, 4]
    y_tr[:, 0] = temp
    temp2 = tr_init[:, 5]
    y_tr[:, 1] = temp2
    # The above 4 lines take the correct values (latitude and longitude) and add them to the response np array

    X_tr = np.empty((len(tr_init), 4)) # Does not take id as a variable
    X_tr_ids = np.array(tr_init[:, 0]).astype(int) # This is a list of ids that we may need later
    for a in range(3): # this weird range plus the end is because of how the dataset was set up
        temp = tr_init[:, a+1]
        X_tr[:, a] = temp
    X_tr[:, 3] = tr_init[:, 6]
    # X_tr is now set up as: (Hour1, Hour2, Hour3, numPosts)

    te_init = loadtxt("posts_test.txt", comments='#', delimiter=",", unpack=False)

    """
    we would need to get the user id too right, in order to correlate that user
    to his social graph?
    """
    # Then, we do the same thing with the test data; start at column 1 and get a 4-column dataset with
    # the columns as hour1, Hour2, Hour3, numposts
    X_te = np.empty((len(te_init), 4))
    X_te_ids = np.array(te_init[:, 0]).astype(int) # This is a list of ids to use later
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
    return set(tr_init[:,0])


def linearRegressor(tr_X, tr_y, te_X):  # Code to play around with for performing a linear regression
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linearRegression.predict(te_X)
    return preds


# The above method uses a simple linear model, which we can edit or make new methods to make more complex,
# and outputs the linear model's prediction

# Below, I'm going to createa n output; we can alter which methods we use when we change which learners we use
y_tr_lat = y_tr[:, 0]
y_tr_long = y_tr[:, 1]
y_lat_preds = np.around(linearRegressor(X_tr, y_tr_lat, X_te), decimals=3)
y_long_preds = np.around(linearRegressor(X_tr, y_tr_long, X_te),
                         decimals=3)  # This creates predictions for lat and long
preds_output = np.ndarray((1000, 3), dtype=object)  # this creates an array that can hold ints and floats
preds_output[:, 0] = X_te_ids
preds_output[:, 1] = y_lat_preds
preds_output[:, 2] = y_long_preds
print(preds_output)
np.savetxt('output1.txt', preds_output, delimiter=',', header="Id,Lat,Lon")  # This saves as a text file


if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    #tr_file = open("posts_train.txt", "r")
    #tr = tr_file.read().split(',')
    #print(tr)
    clean_up_graph(posts_cleanup())

