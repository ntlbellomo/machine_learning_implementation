from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# from decision_tree import DecisionTree
from knn import KNN


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# clf = DecisionTree(max_depth=10)
clf = KNN(k=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, y_pred)

print(acc)
