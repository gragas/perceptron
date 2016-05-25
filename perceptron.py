from random import randint

import numpy as np

class Perceptron:
    def __init__(self, training_data, max_iters=None, threshold=None):
        xs, labels = training_data # xs shape: (num_feature_vectors, num_features)
        self.weights = np.ones(xs.shape[-1] + 1) # weights shape: (num_features + 1, )

        def average_difference(label, y):
            if label.shape:
                s = label.shape[0]
            else:
                s = 1
            return np.sum(np.fabs(label - y)) / s

        if max_iters is None and threshold is None:
            def should_stop(label, feature_vector, weights, curr_iters):
                return average_difference(label, self.predict(feature_vector)) != 0.0
        elif max_iters is not None and threshold is None:
            def should_stop(label, feature_vector, weights, curr_iters):
                return curr_iters >= max_iters
        elif max_iters is None and threshold is not None:
            def should_stop(label, feature_vector, weights, curr_iters):
                return average_difference(label, self.predict(feature_vector)) <= threshold
        else:
            def should_stop(label, feature_vector, weights, curr_iters):
                return curr_iters >= max_iters or \
                    average_difference(label, self.predict(feature_vector)) <= threshold

        curr_iters = 0
        index = 0
        feature_vector = xs[index]
        label = labels[index]
        while not should_stop(label, feature_vector, self.weights, curr_iters):
            diff = (
                np.ones(feature_vector.shape[0] + 1) *  (label - self.predict(feature_vector))
            ) * np.insert(feature_vector, 0, 1)
            self.weights = self.weights + diff
            index = (index + 1) % labels.shape[0]
            feature_vector = xs[index]
            label = labels[index]

    def predict(self, feature_vector):
        total = np.dot(self.weights, np.insert(feature_vector, 0, 1))
        return 1 if total >= 0 else -1

    def predictions(self, xs):
        return np.array(list(map(self.predict, xs)))
