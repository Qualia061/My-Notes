# **_Notes_**

## Some useful ML models

### Logisic Regression

model = LogisticRegression()

### Random Forests Model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

### Support Vector Machines

from sklearn.svm import SVC, LinearSVC

model = SVC()

### Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

### K-nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)

### Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
