# **_Notes_**

## Some useful ML models

### Logisic Regression
```python
model = LogisticRegression()
```
### Random Forests Model
```python
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
