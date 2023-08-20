import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance, function needs samples as cols
        cov = np.cov(X.T)

        # eigenvectors and eigenvalues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # column to row
        eigenvectors = eigenvectors.T

        # sort eigenvectores
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[: self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)