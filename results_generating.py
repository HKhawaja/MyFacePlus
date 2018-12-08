from numpy import loadtxt
import numpy as np
from learners import *
from helpers import *
from sklearn.ensemble import BaggingRegressor

def LinearRegressionOutput(helper):
    # clean_up_graph(posts_cleanup())
    # Below, I'm going to create an output; we can alter which methods we use when we change which learners we use
    data, unused = helper

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
    return preds_output
    # This saves as a text file
    # This actually made an output and finished 8th with a score of 58

def GradientRegressionOutput(helper):
    data, unused = helper # this creates the matrices we want of training and test data
    X_tr = data[0]
    y_tr = data[1]
    X_te = data[2]
    X_te_ids = data[3]
    y_tr_lat = y_tr[:, 0]
    y_tr_long = y_tr[:, 1]
    y_lat_preds = np.around(gradientBooster(X_tr, y_tr_lat, X_te), decimals=3)
    y_long_preds = np.around(gradientBooster(X_tr, y_tr_long, X_te),
                             decimals=3)  # This creates predictions for lat and long
    preds_output = np.ndarray((1000, 3), dtype=object)  # this creates an array that can hold ints and floats
    preds_output[:, 0] = X_te_ids
    preds_output[:, 1] = y_lat_preds
    preds_output[:, 2] = y_long_preds
    return preds_output
    # this is called in main method to create scores

if __name__ == "__main__":
    # assert 1 / 2 == 0.5, "Are you sure you're using python 3?"
    # linear_regression = LinearRegressionOutput(posts_cleanup())
    # np.savetxt('RegressionOutput.txt', linear_regression, delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # # This is linear regression without graphs; gives score of ~58
    #
    # gradient_regression = GradientRegressionOutput(posts_cleanup())
    # np.savetxt('gradientBoostingOutput.txt', gradient_regression,
    #            delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # # This is gradient regression without graphs; gives score of ~46

    medians_gradient_regression = GradientRegressionOutput(posts_cleanup_medians())
    np.savetxt('gradientBoostingOutputMedians.txt', medians_gradient_regression,
               delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    print(medians_gradient_regression)
    # RThis is gradient regression with median lat and lng (max_depth =3); gives score of 25.376
    # Also tested gradient boosting with max depth of 1; gives score of 26.06
    # And with max depth of 5; gives score of 24.88 (THAT'S THE BEST LET'S GO)

    # medians_only_gradient_regression = GradientRegressionOutput(posts_cleanup_only_median())
    # np.savetxt('gradientBoostingOnlyMedians.txt', medians_only_gradient_regression, delimiter=',',
    #            fmt='%1.3f', header="Id,Lat,Lon")
    # # this is a gradient regression using only median lat and lng; gives score of 26.43

    # medians_linear_regression = LinearRegressionOutput(posts_cleanup_medians())
    # np.savetxt('LinearOutputMedians.txt', medians_linear_regression, delimiter=',', fmt='%1.3f',
    #            header="Id,Lat,Lon")
    # # this is a linear regression including median lat and lng; gives score of 26.73

    # medians_only_linear_regression = LinearRegressionOutput(posts_cleanup_only_median())
    # np.savetxt('LinearOutputOnlyMedians.txt', medians_only_linear_regression, delimiter=',', fmt='%1.3f',
    #            header="Id,Lat,Lon")
    # # this is a linear regression using only median lat and lng; gives score of 26.76


    # # added bagging to regression
    # bagged_regression = BaggingRegressor(GradientRegressionOutput(posts_cleanup_only_median()))
    # np.savetxt('bagging_regression.txt', bagged_regression, delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")