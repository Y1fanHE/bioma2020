import numpy as np

def initialize(n_pop, n_var, xl, xu):
    X = np.round(np.random.uniform(xl, xu, (n_pop, n_var)), 14)
    return X

def evaluate(X, f):
    F = np.zeros(X.shape[0])
    for i, x in enumerate(X):
        F[i] = f(x)
    return F