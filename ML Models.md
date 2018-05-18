# **_My Notes_**

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
### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingClassifier()
```
### K-nearest Neighbors
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsClassifier(n_neighbors = 3)
```
### Gaussian Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
```
### XGBoost
```python
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor

model = GaussianNB()
```

### Tuning the Parameters
```python
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

grid.grid_scores_
grid.best_score_    
grid.best_params_  
grid.best_estimator_

#Example
params={'learning_rate':np.linspace(0.05,0.25,5), 'max_depth':[x for x in range(1,8,1)], 'min_samples_leaf':
                [x for x in range(1,5,1)], 'n_estimators':[x for x in range(50,100,10)]}
clf = GradientBoostingClassifier()
grid = GridSearchCV(clf, params, cv=10, scoring="f1")
grid.fit(X, y)
```

## Other Packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import RandomizedSearchCV
```

## Unsolved Problems
1. Data Visualizatoin
2. Overfitting
3. Tuning the parameters

