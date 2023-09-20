# SYSTEM IMPORTS
from abc import abstractmethod, ABC     # you need these to make an abstract class in Python
from typing import List, Type, Union    # Python's typing syntax
import numpy as np      # linear algebra & useful containers   
import math             

# PYTHON PROJECT IMPORTS


# Types defined in this module
DistributionType: Type = Type["Distribution"]
BinomialDistributionType = Type["BinomialDistribution"]
PoissonDistributionType = Type["PoissonDistribution"]
GaussianDistributionType = Type["GaussianDistribution"]

# an abstract class for an arbitrary distribution
# please don't touch this class
class Distribution(ABC):

    # this is how you make an abstract method in Python
    # all child classes MUST implement this method
    # otherwise we will get an error when the child class
    # is instantiated 
    @abstractmethod 
    def fit(self: DistributionType,
            X: np.ndarray               # input data to fit from
            ) -> DistributionType:
        ... # same as "pass"

    # another abstract method that every child class will have to implement
    # in order to be able to be instantiated
    @abstractmethod
    def prob(self: DistributionType,
             X: np.ndarray
             ) -> np.ndarray:           # return Pr[x] for each point (row) of the data (X)
        ... # same as "pass"

    @abstractmethod
    def parameters(self: DistributionType) -> List[Union[float, np.ndarray]]:
        ... # same as "pass"


# a class for the binomial distribution
# you will need to complete this class
class BinomialDistribution(Distribution):
    def __init__(self: BinomialDistributionType,
                 n: int) -> None:
        # controlled by parameter "p"
        self.p: float = None
        self.n: int = n


    def fit(self: BinomialDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> BinomialDistributionType:
                
        for row in X:
            summer = 0.0
            for col in row:
                if col == 1:
                    summer+=1
        summer /= self.n
        self.p = summer
        
        return self # keep this at the end

    def prob(self: BinomialDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        
        prob_list = []
        
        for row in X:
            summer = sum(row)
            length = len(row)
            prob_list += [((self.p ** summer) * ((1-self.p)**(length - summer)))]
        np_Arr = np.array(prob_list).reshape(-1, 1)
        
        return np_Arr
        # TODO: complete me!
        ... # same as "pass"

    def parameters(self: BinomialDistributionType) -> List[Union[float, np.ndarray]]:
        return  [self.n, self.p]


# a class for the poisson distribution
# you will need to complete this class
class PoissonDistribution(Distribution):
    def __init__(self: PoissonDistributionType) -> None:
        # controlled by parameter "lambda"
        self.lamb: float = None

    def fit(self: PoissonDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> PoissonDistributionType:
        summer = 0
        TotalItems = 0
        for row in X:
            summer += sum(row)
            TotalItems += len(row)
        self.lamb = summer / TotalItems
        # TODO: complete me!
        return self # keep this at the end

    def prob(self: PoissonDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        # TODO: complete me!
        ... # same as "pass"
        
        prob_list = []
        
        for row in X:
            summer = sum(row)
            length = len(row)
            prob_list += [(((self.lamb**summer)*(np.exp(-self.lamb)))/(np.math.factorial(summer)))]
        np_Arr = np.array(prob_list).reshape(-1, 1)
        
        return np_Arr

    def parameters(self: PoissonDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.lamb]


class GaussianDistribution(Distribution):
    def __init__(self: GaussianDistributionType) -> None:
        # controlled by parameters mu and var
        self.mu: float = None
        self.var: float = None

    def fit(self: GaussianDistributionType,
            X: np.ndarray                   # input data to fit from
                                            # this will be a bunch of integer samples stored in a column vector
            ) -> GaussianDistributionType:
        self.var = np.var(X)
        self.mu = np.mean(X)
        # TODO: complete me!
        return self # keep this at the end

    def prob(self: GaussianDistributionType,
             X: np.ndarray                  # this will be a column vector where every element is a float
             ) -> np.ndarray:
        
        prob_list = []
        
        for row in X:
            summer = sum(row)
            length = len(row)
            x = summer / length
            #x=row[0]
            prob_list += [((1)/(math.sqrt((2*math.pi*self.var))))*np.exp(-(((x-self.mu)**2)/(2*self.var)))]
            #[((1)/(math.sqrt(2*math.pi*self.var)))*(np.exp((-((x-self.mu)**2)/(2*self.var))))] 
        np_Arr = np.array(prob_list).reshape(-1, 1)
        
        return np_Arr
        
        # TODO: complete me!
        ... # same as "pass"

    def parameters(self: GaussianDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.mu, self.var]


