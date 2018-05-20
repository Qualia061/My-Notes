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
1. Set initial values of parameters
```python
xgb_best = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
```
2.
