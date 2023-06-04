
// XGboost code for classification

import xgboost as xgb

import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix 

# load data

iris = load_iris()

X = iris.data

y = iris.target

# split data into train and test sets

seed = 7

test_size = 0.33

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

# fit model no training data

model = xgb.XGBClassifier()

model.fit(X_train, y_train)

# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# confusion matrix

print(confusion_matrix(y_test, y_pred))


# Path: dailynlp/py0002/example.py


