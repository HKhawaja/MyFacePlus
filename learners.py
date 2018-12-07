import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def linearRegressor(tr_X, tr_y, te_X):  # Code to play around with for performing a linear regression
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linear_tr.predict(te_X)
    return preds

def gradientBooster(tr_X, tr_y, te_X):
    GradientRegression = GradientBoostingRegressor()
    gradientFit = GradientRegression.fit(X=tr_X, y=tr_y, sample_weight=None)
    gradient_preds = gradientFit.predict(te_X)
    return gradient_preds
