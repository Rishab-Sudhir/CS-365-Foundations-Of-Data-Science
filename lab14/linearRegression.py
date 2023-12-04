# SYSTEM IMPORTS
from typing import Callable, List, Type, Tuple                  # typing info
from sklearn.datasets import load_diabetes                      # regression dataset
from sklearn.metrics import mean_squared_error                  # our performance metric
from tqdm import tqdm                                           # progress bar in python
import matplotlib.pyplot as plt                                 # drawing stuff in python
import numpy as np                                              # linear algebra
import os                                                       # manipulating paths


# PYTHON PROJECT IMPORTS


# CONSTANTS


# TYPES DEFINED
LinearRegressionType = Type["LinearRegression"]


 def fit(self: LinearRegressionType,
            X: np.ndarray,
            y_gt: np.ndarray
            ) -> LinearRegressionType:

        # TODO: complete me!
        # This method should populate self.weights

         #Pad X with a row of 1's
        n_rows, n_cols = X.shape
        X_pad = np.hstack([X, np.ones((n_rows, 1), dtype=float)])

        #Compute weights
        pseudo_inv = np.linalg.pinv(X_pad)
        self.weights = pseudo_inv @ y_gt

        # don't change the return type!
        return self

    def predict(self: LinearRegressionType,
                X: np.ndarray
                ) -> np.ndarray:

        # TODO: complete me!
        # This method should use the self.weights
        # you set to make predictions on X!


        slope_part = X @ self.weights[:-1, :]
        intercept_part = self.weights[-1]
        pred = slope_part +intercept_part
        return pred



def test_1d(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(12345) # make results reproducible

    true_coeff: float = 3.0
    true_offset: float = 1.5

    X: np.ndarray = np.arange(-15, 15, 0.2).reshape(-1, 1)
    y_gt: np.ndarray = true_coeff * X + true_offset #  + np.random.randn(*X.shape)

    m: LinearRegression = LinearRegression().fit(X, y_gt)

    if abs(true_coeff - m.weights[0, 0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[0, 0] should contain coeff close to {true_coeff}")
    if abs(true_offset - m.weights[1, 0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[1, 0] should contain offset close to {true_offset}")

    print("test_1d PASSED")


def test_2d(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(12345) # make results reproducible

    true_coeffs: np.ndarray = np.array([3.0, -2.5]).reshape(-1,1)
    true_offset: float = 1.5

    X: np.ndarray = np.random.randn(num_samples, 2)
    y_gt: np.ndarray = X.dot(true_coeffs) + true_offset #  + np.random.randn(*X.shape)

    m: LinearRegression = LinearRegression().fit(X, y_gt)

    if abs(true_coeffs[0,0] - m.weights[0,0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[0, 0] should contain coeff close to {true_coeffs[0,0]}")
    if abs(true_coeffs[1,0] - m.weights[1,0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[0, 0] should contain coeff close to {true_coeffs[1,0]}")
    if abs(true_offset - m.weights[2, 0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[1, 0] should contain offset close to {true_offset}")

    print("test_2d PASSED")


def test_nd(num_samples: int,
            num_dims: int
            ) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(12345) # make results reproducible

    true_coeffs: np.ndarray = np.random.randn(num_dims, 1)
    true_offset: float = np.random.randn()

    X: np.ndarray = np.random.randn(num_samples, num_dims)
    y_gt: np.ndarray = X.dot(true_coeffs) + true_offset #  + np.random.randn(*X.shape)

    m: LinearRegression = LinearRegression().fit(X, y_gt)

    # print(true_coeffs.reshape(-1), m.weights.reshape(-1))

    for idx in range(num_dims):
        if abs(true_coeffs[idx,0] - m.weights[idx,0]) > 1e-8:
            raise Exception(f"ERROR: m.weights[{idx}, 0] should contain coeff close to {true_coeffs[idx,0]}")
    if abs(true_offset - m.weights[-1, 0]) > 1e-8:
        raise Exception(f"ERROR: m.weights[1, 0] should contain offset close to {true_offset}")

    print(f"test_{num_dims}d PASSED")


def main() -> None:

    test_1d(1000)
    test_2d(1000)
    for num_dims in range(3, 15):
        test_nd(1000, num_dims)

if __name__ == "__main__":
    main()

