# SYSTEM IMPORTS
from typing import Callable, List, Type, Tuple          # typing info
from tqdm import tqdm                                   # progress bar in python
import numpy as np                                      # linear algebra
import os                                               # manipulating paths


# PYTHON PROJECT IMPORTS



# CONSTANTS
CD: str = os.path.abspath(os.path.dirname(__file__))    # get dir to this file
DATA_DIR: str = os.path.join(CD, "data")                # make path relative to this file
CLASS_0_DIR: str = os.path.join(DATA_DIR, "class_0")
CLASS_1_DIR: str = os.path.join(DATA_DIR, "class_1")


# TYPES DEFINED



def load_data() -> Tuple[np.ndarray, np.ndarray]:
    class_0_data: List[np.ndarray] = list()
    class_1_data: List[np.ndarray] = list()

    # load in all the individual numpy arrays and flatten them
    for list_to_populate, path_to_load in zip([class_0_data, class_1_data],
                                              [CLASS_0_DIR,  CLASS_1_DIR]):
        for npy_file in [x for x in os.listdir(path_to_load)
                         if x.endswith(".npy")]:
            list_to_populate.append(np.load(os.path.join(path_to_load, npy_file)).reshape(-1))

    # make a matrix from each list, where each element of the list is a row
    # don't forget to change this into floats!
    return tuple([np.vstack(data_list).astype(float)
                  for data_list in [class_0_data, class_1_data]])


def check_2d(X: np.ndarray) -> None:
    if len(X.shape) != 2:
        raise ValueError(f"ERROR: expected X to be 2d but had shape {X.shape}!")


def check_same_num_examples(X: np.ndarray,
                            Y: np.ndarray
                            ) -> bool:
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"ERROR: expected X & Y to have same # of rows: X={X.shape}, Y={Y.shape}")


def randomly_project(X: np.ndarray,                     # the original dataset
                     k: int                             # the dimensionality to reduce to
                     ) -> Tuple[np.ndarray, np.ndarray]:
    # this function should return a pair:
    #   (f, X_reduced)
    check_2d(X)
    d = X.shape[1] #returns the number of columns and therefore the number of dimensions
    A = np.random.normal(loc=0,scale = 1, size=(d,k)) #Creating a randomly populated normal matrix 
    #Loc = mean , Scale = Standard Deviation,  size = (rows, columns)
    A = A/np.sqrt(k) #Function f being applied to A
    # X is n x d and A is d x k, Doing X @ A returns a reduced Matrix of size d x k
    X_reduced = X @ A
    
    return A, X_reduced
    
    # TODO: complete me!


def check_if_distance_satisfied(X: np.ndarray,          # the original dataset
                                X_reduced: np.ndarray,  # the reduced dataset
                                epsilon: float          # how far away points can be without breaking constraints
                                ) -> bool:
    # this function should return False if any of the points in X break a constraint
    # and True otherwise
    check_2d(X)
    check_2d(X_reduced)
    check_same_num_examples(X, X_reduced)
    
    n = X.shape[0] #returns the number of rows and therefore the number of data points
    
    for i in range(n):
        for j in range(i + 1, n):  # Avoid duplicate pairs and self-comparison
            # Compute the Euclidean distance between x_i and x_j in the original space
            dist_in_og_space = np.linalg.norm(X[i, :] - X[j, :], ord=2)
            
            # Compute the Euclidean distance between x_i and x_j in the reduced space
            dist_in_reduced_space = np.linalg.norm(X_reduced[i, :] - X_reduced[j, :], ord=2)
            
            # Check the Johnson-Lindenstrauss lemma conditions
            if not (((1 - epsilon) * dist_in_og_space <= dist_in_reduced_space) and (dist_in_reduced_space <= (1 + epsilon) * dist_in_og_space)):
                return False
    
    return True
    # TODO: complete me!


def reduce_dims_randomly(X: np.ndarray,                 # the original dataset
                         k: int,                        # the dimensionality to reduce to
                         epsilon: float                 # how far away points can be without breaking constraints
                         ) -> Tuple[np.ndarray, np.ndarray, int]:
    # this function should return a triple:
    #   (f, X_reduced, num_iterations)

    # TODO: complete me!
    
    num_iter = 0
    stop = False
    
    #While the Johnson-Lindenstrauss lemma is not true repeat
    while not stop:
        A, X_reduced = randomly_project(X, k)
        #Create a random projection, and use it to create the reduced matrix
        
        stop = check_if_distance_satisfied(X, X_reduced, epsilon)
        #check if the geometry of the reduced matrix is the same as the original
        
        num_iter += 1
    
    return A, X_reduced, num_iter 
    ...


def main() -> None:
    X_class_0, X_class_1 = load_data()
    print([X.shape for X in [X_class_0, X_class_1]])

    X: np.ndarray = np.vstack([X_class_0, X_class_1])
    print(X.shape)


    # if we find 1000 random projections, the average number of iterations
    # should converge to at most the expected number of iterations
    # which we can calculate knowing that the number of iterations is a geometric random variable.
    # what is the probability of success?
    num_samples: int = 1000
    iter_samples: List[int] = list()

    for _ in tqdm(range(num_samples)):
        _, _, num_iter = reduce_dims_randomly(X, 32, 0.3)
        iter_samples.append(num_iter)

    print("avg number of iterations=", np.mean(iter_samples))


if __name__ == "__main__":
    main()

