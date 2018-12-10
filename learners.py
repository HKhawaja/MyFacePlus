import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

def linearRegressor(tr_X, tr_y, te_X):  # Code for performing a linear regressor
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linear_tr.predict(te_X)
    return preds

def gradientBooster(tr_X, tr_y, te_X): #Code for performing a gradient boosting regressor
    GradientRegression = GradientBoostingRegressor(max_depth=5) # this can be changed to alter results
    gradientFit = GradientRegression.fit(X=tr_X, y=tr_y, sample_weight=None)
    gradient_preds = gradientFit.predict(te_X)
    return gradient_preds
