from learners import *
from helpers import *

def LinearRegressionOutput(helper):
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
    # this outputs the predictions for a linear regression with parameters defined in learners.py

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

def AdaBoostOutput(helper):
    data, unused = helper # this creates the matrices we want of training and test data
    X_tr = data[0]
    y_tr = data[1]
    X_te = data[2]
    X_te_ids = data[3]
    y_tr_lat = y_tr[:, 0]
    y_tr_long = y_tr[:, 1]
    y_lat_preds = np.around(ada_boost_regressor(X_tr, y_tr_lat, X_te), decimals=3)
    y_long_preds = np.around(ada_boost_regressor(X_tr, y_tr_long, X_te),
                             decimals=3)  # This creates predictions for lat and long
    preds_output = np.ndarray((1000, 3), dtype=object)  # this creates an array that can hold ints and floats
    preds_output[:, 0] = X_te_ids
    preds_output[:, 1] = y_lat_preds
    preds_output[:, 2] = y_long_preds
    return preds_output
    # this is called in main method to create scores

def BagRegressorOutput(helper):
    data, unused = helper # this creates the matrices we want of training and test data
    X_tr = data[0]
    y_tr = data[1]
    X_te = data[2]
    X_te_ids = data[3]
    y_tr_lat = y_tr[:, 0]
    y_tr_long = y_tr[:, 1]
    y_lat_preds = np.around(bagging_regressor(X_tr, y_tr_lat, X_te), decimals=3)
    y_long_preds = np.around(bagging_regressor(X_tr, y_tr_long, X_te),
                             decimals=3)  # This creates predictions for lat and long
    preds_output = np.ndarray((1000, 3), dtype=object)  # this creates an array that can hold ints and floats
    preds_output[:, 0] = X_te_ids
    preds_output[:, 1] = y_lat_preds
    preds_output[:, 2] = y_long_preds
    return preds_output
    # this is called in main method to create scores


if __name__ == "__main__":
    medians_gradient_regression = GradientRegressionOutput(posts_cleanup_medians())
    np.savetxt('gradientBoostingOutputMedians.txt', medians_gradient_regression,
               delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # Best performing depth found after CV is max_depth = 5 on lat and max_depth = 6 on longitudes
    # with score on test set being 24.88

    # medians_gradient_regression = BagRegressorOutput(posts_cleanup_medians())
    # np.savetxt('BagRegressorOutputMedians.txt', medians_gradient_regression,
    #            delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # Best parameters for Bagged Decision Trees were {'n_estimators':20} for both, lat and lng
    # Test score: 26.67

    # medians_gradient_regression = AdaBoostOutput(posts_cleanup_medians())
    # np.savetxt('adaBoostingOutputMedians.txt', medians_gradient_regression,
    #            delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # Best hyperparameters for AdaBoost were {'n_estimators': 100, 'learning_rate': 0.01} for lat and
    # {'n_estimators': 50, 'learning_rate': 0.01} for long
    # Test score: 26.77

    # medians_only_linear_regression = LinearRegressionOutput(posts_cleanup_only_median())
    # np.savetxt('LinearOutputOnlyMedians.txt', medians_only_linear_regression, delimiter=',', fmt='%1.3f',
    #            header="Id,Lat,Lon")
    # this is a linear regression using only median lat and lng; gives score of 26.76

    # linear_regression = LinearRegressionOutput(posts_cleanup())
    # np.savetxt('RegressionOutput.txt', linear_regression, delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # This is linear regression without graphs; gives score of ~58

    # gradient_regression = GradientRegressionOutput(posts_cleanup())
    # np.savetxt('gradientBoostingOutput.txt', gradient_regression,
    #            delimiter=',', fmt='%1.3f', header="Id,Lat,Lon")
    # This is gradient regression without graphs; gives score of ~46

    # medians_only_gradient_regression = GradientRegressionOutput(posts_cleanup_only_median())
    # np.savetxt('gradientBoostingOnlyMedians.txt', medians_only_gradient_regression, delimiter=',',
    #            fmt='%1.3f', header="Id,Lat,Lon")
    # this is a gradient regression using only median lat and lng; gives score of 26.43

    # medians_linear_regression = LinearRegressionOutput(posts_cleanup_medians())
    # np.savetxt('LinearOutputMedians.txt', medians_linear_regression, delimiter=',', fmt='%1.3f',
    #            header="Id,Lat,Lon")
    # this is a linear regression including median lat and lng; gives score of 26.73

