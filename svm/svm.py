import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        # init weight

        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weight) - self.bias) >= 1
                if condition:
                    self.weight -= self.learning_rate * (
                        2 * self.lambda_param * self.weight
                    )
                else:
                    self.weight -= self.learning_rate * (
                        2 * self.lambda_param * self.weight - np.dot(x_i, y_[idx])
                    )
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.weight) - self.bias
        return np.sign(approx)
