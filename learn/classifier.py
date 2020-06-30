import numpy as np
import math
import random
import operator
from sklearn.naive_bayes import GaussianNB


# --------------------------------------------------------------------------------
#                          Naive Bayes Classifier
# --------------------------------------------------------------------------------

class Func:
    def __init__(self, rel, label):
        self.rel = rel
        self.label = label

    def func(self, x):
        if self.rel == "==":
            return self.label == x
        elif self.rel == ">":
            return x > self.label
        elif self.rel == ">=":
            return x >= self.label
        elif self.rel == "<":
            return x < self.label
        elif self.rel == "<=":
            return x <= self.label

    def __repr__(self):
        return str(self.rel) + str(self.label)


class NaiveBayes:

    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        y_pred = []
        X = np.array(X)
        m, n = self.X.shape
        k = m
        l = int(math.sqrt(n))

        nb = GaussianNB()
        nb.row_idxs = np.random.choice(m, k)
        nb.col_idxs = list(set(np.random.choice(n, l)))
        nb.fit(self.X, self.y)
        for xi in X:
            predictions = {}
        predicted = nb.predict(xi.reshape(1, -1))[0]
        predictions[predicted] = predictions.get(predicted, 0) + 1
        sorted_votes = list(predictions.items())
        sorted_votes.sort(key=operator.itemgetter(1))
        y_pred.append(sorted_votes[-1][0])
        return y_pred

# --------------------------------------------------------------------------------
