# **_Parameter Tuning for ML Models_**

## Some Popular ML Models

### GradientBoosting
```python

```
### Random Forests Model
```python

```
### XGBoost
Import libraries
```python
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV
```
Step 1: Set initial values of parameters
```python
xgb_best = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic', # or multi:softmax, reg:linear
 nthread=4,
 scale_pos_weight=1,
 seed=27)
```
Step 2: Tune max_depth and min_child_weight
```python
param_test1 = {
 'max_depth':[x for x in range(3,10,2)],
 'min_child_weight':[x for x in range(1,6,2)]
}

grid = GridSearchCV(estimator = xgb_best, param_grid = param_test1, cv=5)
grid.fit( source_X , source_y )
grid.grid_scores_
grid.best_estimator_
```
Step 3: Tune gamma
```python
param_test2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
```
Step 4: Tune subsample and colsample_bytree
```python
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
```
Step 5: Tuning Regularization Parameters
```python
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
```
Step 6: Reducing Learning Rate
