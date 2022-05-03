import numpy as np
from numpy.linalg import inv
from MatrixOperations import Matrix


class MyLinearBivariateRegression:

    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []
        self.matrix = Matrix()

    def fit(self, input, output):
        output = [[el] for el in output]
        X = np.array(input)
        Y = np.array(output)

        ones = np.ones(shape=Y.shape)

        X = np.concatenate((ones, X), 1)

        coefficients = inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)

        coefficients = self.matrix.transpose_matrix(coefficients)
        self.intercept_ = coefficients[0][0]
        self.coef_ = coefficients[0][1:]

        return self.intercept_, self.coef_

    def predict(self, input):
        X = np.array(input)
        ones = np.ones(shape=(X.shape[0], 1))
        X = np.concatenate((ones, X), 1)
        coefficients = [self.intercept_] + [c for c in self.coef_]
        coefficients = np.array(coefficients)
        coefficients = coefficients.transpose()
        result = X.dot(coefficients)
        return result
