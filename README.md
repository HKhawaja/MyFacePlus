# MyFacePlus
Project 4 for Machine Learning Course

# How to Run
Navigate to directory and run
 
`pip install -r requirements.txt`

`python3 results_generating.py`

Results should be outputted in a txt file called `gradientBoostingOutputMedians.txt`

In the outputted txt file, remove the `#` in the first line and
round ids (first column) to integers.  

# Code description
The code contains 3 files: `helpers.py`, `learners.py` and 'results_generating.py'

#Learners.py
Contains 2 regressors that take X_tr, y_tr and X_te and output predictions. 

#Helpers.py
Data processing, feature extraction from graph and creation of training and test sets

#Results_generating.py
Takes in learner and processed data and outputs final predictions from its main method.  

 