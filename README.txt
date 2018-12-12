# MyFacePlus
Project 4 for Machine Learning Course

# How to Run
Navigate to directory and run

`pip install -r requirements.txt`

`python3 results_generating.py`

Results for the best learner should be outputted in a txt file called `gradientBoostingOutputMedians.txt`

In the outputted txt file, remove the `#` in the first line and
round ids (first column) to integers [replace .000 with empty space]

# Code description
The code contains 3 files: `helpers.py`, `learners.py` and 'results_generating.py'

#Learners.py
Contains regressors that take X_tr, y_tr and X_te and output predictions. X_tr and X_te are scaled.
The regressors we used are: LinearRegressor and some ensemble learners
(BaggingRegressor, GradientBoostingRegressor and AdaBoost Regressor).
We perform a grid search using cross validation for the best hyperparameters before predicting.

#Helpers.py
Data processing, feature extraction from graph and creation of training and test sets.

#Results_generating.py
Runs the best learner we found (Gradient Boosting Regressor) and outputs the predictions
into a .txt file.
