from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

from naive_bayes import NaiveBayes


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = NaiveBayes()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)


acc = accuracy(y_test, y_pred)

print(acc)
