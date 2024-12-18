import numba
import numpy as np
import math
import random
from typing import List

@numba.jit(nopython=True, fastmath=False, cache=True)
def numba_propagate(sample_inputs, X, d, L, W, is_classification):
    assert (len(sample_inputs) == d[0])

    for i in range(1, d[0] + 1):
        X[0, i] = sample_inputs[i - 1]

    for l in range(1, L + 1):
        for j in range(1, d[l] + 1):
            total = 0.0
            for i in range(0, d[l - 1] + 1):
                total += W[l, i, j] * X[l - 1, i]

            if is_classification or l < L:
                total = math.tanh(total)

            X[l, j] = total


@numba.jit(nopython=True, fastmath=False, cache=True)
def numba_train_step(sample_inputs, sample_expected_outputs,
                     X, d, L, W, is_classification, deltas, alpha):
    numba_propagate(sample_inputs, X, d, L, W, is_classification)

    for j in range(1, d[L] + 1):
        deltas[L, j] = (X[L, j] - sample_expected_outputs[j - 1])

        if is_classification:
            deltas[L, j] *= (1 - X[L, j] ** 2)

    for rev_l in range(0, L - 1):
        l = L - rev_l
        for i in range(1, d[l - 1] + 1):
            total = 0.0
            for j in range(1, d[l] + 1):
                total += W[l, i, j] * deltas[l, j]

            total *= (1.0 - X[l - 1, i] ** 2)
            deltas[l - 1, i] = total

    for l in range(1, L + 1):
        for i in range(0, d[l - 1] + 1):
            for j in range(1, d[l] + 1):
                W[l, i, j] -= alpha * X[l - 1, i] * deltas[l, j]

@numba.jit(nopython=True, fastmath=False, cache=True)
def numba_train(nb_iter, all_samples_inputs, all_samples_expected_outputs,
                X, d, L, W, is_classification, deltas, alpha):
    for it in range(nb_iter):
        k = random.randint(0, len(all_samples_inputs) - 1)
        sample_inputs = all_samples_inputs[k]
        sample_expected_outputs = all_samples_expected_outputs[k]

        numba_train_step(sample_inputs, sample_expected_outputs,
                         X, d, L, W, is_classification, deltas, alpha)


class MLP:
    def __init__(self, npl: List[int]):
        self.d = np.array(list(npl))
        max_neurons = np.max(self.d) + 1
        self.W = np.zeros((len(self.d), max_neurons, max_neurons))
        self.X = np.zeros((len(self.d), max_neurons))
        self.deltas = np.zeros((len(self.d), max_neurons))
        self.L = len(npl) - 1

        for l in range(0, self.L + 1):
            if l == 0:
                continue

            for i in range(0, self.d[l - 1] + 1):

                for j in range(0, self.d[l] + 1):
                    if j == 0:
                        self.W[l, i, j] = 0.0
                    else:
                        self.W[l, i, j] = random.random() * 2.0 - 1.0

        for l in range(0, self.L + 1):
            for j in range(0, self.d[l] + 1):
                self.deltas[l, j] = 0.0
                self.X[l, j] = 1.0 if j == 0 else 0.0

    def _propagate(self, sample_inputs: np.ndarray, is_classification: bool = True):
        numba_propagate(sample_inputs, self.X, self.d, self.L, self.W, is_classification)

    def predict(self, sample_inputs: np.ndarray, is_classification: bool = True) -> np.ndarray:
        self._propagate(sample_inputs, is_classification)
        return self.X[self.L, 1:self.d[self.L] + 1]

    def train(self,
              all_samples_inputs: List[List[float]],
              all_samples_expected_outputs: List[List[float]],
              alpha: float = 0.1,
              nb_iter: int = 10000,
              is_classification: bool = True
              ):
        numba_train(nb_iter, all_samples_inputs, all_samples_expected_outputs,
                    self.X, self.d, self.L, self.W, is_classification, self.deltas, alpha)
