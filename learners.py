from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def linearRegressor(tr_X, tr_y, te_X):  # Code for performing a linear regressor
    linearRegression = LinearRegression(normalize=True)  # creates a linear regression object
    linear_tr = linearRegression.fit(tr_X, tr_y)
    preds = linear_tr.predict(te_X)
    return preds

def gradientBooster(tr_X, tr_y, te_X): #Code for performing a gradient boosting regressor

    # scale the training data and use the same scaler to transform both, training and test data
    scaler = MinMaxScaler()
    scaler.fit(tr_X)
    tr_X = scaler.transform(tr_X)

    # run cross validation to find best depth
    hyper_parameters = [
        {
            'max_depth': [1,2,3,4,5,6,7,8,9,10]
        }
    ]

    model = GridSearchCV(GradientBoostingRegressor(), hyper_parameters, cv=5, n_jobs=-2)
    model.fit(tr_X, tr_y)

    # gradientFit = GradientRegression.fit(X=tr_X, y=tr_y, sample_weight=None)
    te_X = scaler.transform(te_X)
    preds = model.predict(te_X)
    print('Best learner found: ', end='')
    print(model.best_params_)
    return preds

def ada_boost_regressor(tr_X, tr_y, te_X):

    # scale the training data and use the same scaler to transform both, training and test data
    scaler = MinMaxScaler()
    scaler.fit(tr_X)
    tr_X = scaler.transform(tr_X)

    # run cross validation to find best depth
    hyper_parameters = [
        {
            'n_estimators': [50],
            'learning_rate': [0.01,0.05,0.1,0.3,1]
        },
        {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        }
    ]

    model = GridSearchCV(AdaBoostRegressor(), hyper_parameters, cv=5, n_jobs=-2)
    model.fit(tr_X, tr_y)

    # gradientFit = GradientRegression.fit(X=tr_X, y=tr_y, sample_weight=None)
    te_X = scaler.transform(te_X)
    preds = model.predict(te_X)
    print('Best learner found: ', end='')
    print(model.best_params_)
    return preds

def bagging_regressor(tr_X, tr_y, te_X):
    # scale the training data and use the same scaler to transform both, training and test data
    scaler = MinMaxScaler()
    scaler.fit(tr_X)
    tr_X = scaler.transform(tr_X)

    # run cross validation to find best depth
    hyper_parameters = [
        {
            'n_estimators': [5,10,15,20]
        }
    ]

    model = GridSearchCV(BaggingRegressor(), hyper_parameters, cv=5, n_jobs=-2)
    model.fit(tr_X, tr_y)

    # gradientFit = GradientRegression.fit(X=tr_X, y=tr_y, sample_weight=None)
    te_X = scaler.transform(te_X)
    preds = model.predict(te_X)
    print('Best learner found: ', end='')
    print(model.best_params_)
    return preds