from numpy import loadtxt
import numpy as np
from learners import *
from helpers import *

def LinearRegressionOutput():
    # clean_up_graph(posts_cleanup())
    # Below, I'm going to create an output; we can alter which methods we use when we change which learners we use
    data = posts_cleanup()

    X_tr = data[0]
    y_tr = data[1]
    X_te = data[2]
    X_te_ids = data[3]
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
    np.savetxt('output.txt', preds_output, delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # This saves as a text file
    # This actually made an output and finished 8th with a score of 58

if __name__ == "__main__":
    assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    LinearRegressionOutput()