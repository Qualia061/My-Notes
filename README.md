# **_Notes_**

## Some Popular ML Models

### Logisic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)
prediction=model.predict(pred_x)
```
### Random Forests Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
```
### Support Vector Machines
```python
from sklearn.svm import SVC, LinearSVC

model = SVC()
```
### Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingClassifier()
```
### K-nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)
```
### Gaussian Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
```

## Other Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
```
