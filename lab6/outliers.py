# SYSTEM IMPORTS
from typing import Callable, List, Type, Tuple                  # typing info
from tqdm import tqdm                                           # progress bar in python
import matplotlib.pyplot as plt                                 # drawing stuff in python
import numpy as np                                              # linear algebra
import os                                                       # manipulating paths


# PYTHON PROJECT IMPORTS



# CONSTANTS
CD: str = os.path.abspath(os.path.dirname(__file__))            # get dir to this file
DATA_DIR: str = os.path.join(CD, "data")                        # make path relative to this file
DATA_FILEPATH: str = os.path.join(DATA_DIR, "spatial_data.txt") # actual filepath



K: int = 10                                                   # TODO: Set this after looking at the plot!
ZSCORE_STDDEV_THRESHOLD: float = 2                           # TODO: Set this!


# TYPES DEFINED



def load_data() -> np.ndarray:
    return np.loadtxt(DATA_FILEPATH)


def convert_sigmas_to_info_retained(X: np.ndarray
                                    ) -> np.ndarray:

    # get the SVD of X
    U, sigma_diag, V_t = np.linalg.svd(X, full_matrices=False)

    # first get the norm of X
    X_frobenius_norm: float = np.linalg.norm(X.reshape(-1), ord=2)

    info_retained: List[float] = [0.0] # initially no info retained

    # try to project the data into 1d, 2d, 3d, ...., no projection
    for k in tqdm(range(1, sigma_diag.shape[0]),
                  desc="calculating % info retained per number of new features kept in svd"):

        # keep the first k singular value (set the rest to 0)....this "deletes" them
        sigma_diag_copy: np.ndarray = np.copy(sigma_diag)
        sigma_diag_copy[k:] = 0

        # reconstruct X using the SVD of X but only from the first k new features
        # this reconstruction is expressed in the original feature space
        # but only has info from the first k new features
        X_reconstructed: np.ndarray = U.dot(np.diag(sigma_diag_copy)).dot(V_t)

        # measure how far away the reconstruction is from X as a percentage of the norm of X
        info_retained.append(1 - (np.linalg.norm(X_reconstructed - X, ord=2) / X_frobenius_norm))

    # turn list into a numpy array
    return np.array(info_retained)



def reconstruct_X_using_k_features(X: np.ndarray,
                                   k: int
                                   ) -> np.ndarray:
    # get the SVD of X
    U, sigma_diag, V_t = np.linalg.svd(X, full_matrices=False)
    sigma_diag[k:] = 0
    return U @ np.diag(sigma_diag) @ V_t #np.diag(sigma_diag) converts a 1-D array to a nxn diagonal matrix

    # TODO: complete me!


def calculate_distances(X: np.ndarray,
                        X_reconstructed: np.ndarray
                        ) -> np.ndarray:
    # TODO: complete me!
    
    return np.linalg.norm(X-X_reconstructed, ord=2, axis = 1)
    # Axis defines the axis along which we sum, look at lab 6 hand written notes for more explanations
    ...


def z_score(distances: np.ndarray) -> np.ndarray:
    # TODO: complete me!
    ...
    return (distances - np.mean(distances))/np.std(distances)
     # calculating z scores
     
def get_outlier_idxs(z_scores: np.ndarray,
                     threshold: float
                     ) -> np.ndarray:
    mask = np.abs(z_scores) > threshold
    #mask is a bolean array that contains true/false values for each value in the z_scores np array
    return np.argwhere(mask)
    #np.argwhere returns the indicies of any non_zero values in the input array: boolean in python(false/true) equate to 0/1
    # TODO: complete me!
    ...


def main() -> None:
    X: np.ndarray = load_data()

    # after you use this plot to set K, feel free to comment out this code
    # as it takes a min to run
    info_retained_per_new_feature: np.ndarray = convert_sigmas_to_info_retained(X)
    plt.plot(info_retained_per_new_feature)
    plt.ylabel("% of original info retained in SVD")
    plt.xlabel("number of new features kept")
    plt.show()


    # use K to reconstruct X (this is the 1% use case for SVD)
    X_reconstructed: np.ndarray = reconstruct_X_using_k_features(X, K)

    # measure how far away each point is from its reconstruction (using euclidean distance)
    point_l2_distances: np.ndarray = calculate_distances(X, X_reconstructed)

    # convert these distance samples into z-scores
    point_z_scores: np.ndarray = z_score(point_l2_distances)

    # get the outliers that are X far away (thats the nice part of using z scores)
    outlier_idxs: np.ndarray = get_outlier_idxs(point_z_scores, ZSCORE_STDDEV_THRESHOLD)

    # plot the points
    plt.scatter(np.arange(point_z_scores.shape[0]), point_z_scores, label="'normal' points")
    plt.plot(np.ones_like(point_z_scores) * ZSCORE_STDDEV_THRESHOLD, color="r", label="+ outlier threshold")
    plt.plot(np.ones_like(point_z_scores) * -ZSCORE_STDDEV_THRESHOLD, color="r", label="- outlier threshold")
    plt.scatter(outlier_idxs, point_z_scores[outlier_idxs], label="'outlier' points")
    plt.ylabel("z score")
    plt.xlabel("pt")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

