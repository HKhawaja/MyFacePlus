import sklearn
from sklearn.linear_model import LinearRegression

def linearRegressor(tr_X, tr_y, te_X):  # Code to play around with for performing a linear regression
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linearRegression.predict(te_X)
    return preds