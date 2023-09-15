# SYSTEM IMPORTS
from collections import defaultdict
from typing import Dict, List, Tuple, Type
import os
import sys
from pprint import pprint


_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from data import Color, Softness, GoodToEat, load_data



# TYPES DEFINED IN THIS MODULE
AvacadoPredictorType: Type = Type["AvacadoPredictor"]

def normalize_pmf(dict_to_be_pmf: Dict[str, float]) -> None:
    
    total: int = sum(dict_to_be_pmf.values())
    
    for key in dict_to_be_pmf.keys():
        dict_to_be_pmf[key] /= total


class AvacadoPredictor(object):
    def __init__(self: AvacadoPredictorType) -> None:
        self.color_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Color, float]] = defaultdict(lambda: defaultdict(float))
        self.softness_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Softness, float]] = defaultdict(lambda: defaultdict(float))
        self.good_to_eat_prior: Dict[GoodToEat, float] = defaultdict(float)


    def fit(self: AvacadoPredictorType,
            data: List[Tuple[Color, Softness, GoodToEat]]
            ) -> AvacadoPredictorType: 
        
        # goodToEat_count = sum(1 for _, _, good in data if good == GoodToEat.YES)
        
        # TotalItems = len(data)
        
        # self.good_to_eat_prior[GoodToEat.YES] = goodToEat_count/TotalItems
        # self.good_to_eat_prior[GoodToEat.NO] =  1 - (goodToEat_count/TotalItems)
        
        # color_good_to_eat_counts = {color: 0 for color in Color}
        
        # for color, _, good_to_eat in data:
        #     if good_to_eat == GoodToEat.YES:
        #             color_good_to_eat_counts[color] += 1
        
        # for x in Color:
        #     self.color_given_good_to_eat_pmf[GoodToEat.YES][x] = color_good_to_eat_counts[x] / goodToEat_count
        #     self.color_given_good_to_eat_pmf[GoodToEat.NO][x] = 1 - ( color_good_to_eat_counts[x] / goodToEat_count)
          
        # softness_good_to_eat_counts = {softness: 0 for softness in Softness}
            
        # for _, softness, good_to_eat in data:
        #     if good_to_eat == GoodToEat.YES:
        #             softness_good_to_eat_counts[softness] += 1
            
        #     for x in Softness:
        #         self.softness_given_good_to_eat_pmf[GoodToEat.YES][x] = softness_good_to_eat_counts[x] / goodToEat_count
        #         self.softness_given_good_to_eat_pmf[GoodToEat.NO][x] = 1 - ( softness_good_to_eat_counts[x] / goodToEat_count)
                
        for (color, softness, good_to_eat) in data:
            self.good_to_eat_prior[good_to_eat] += 1
            self.color_given_good_to_eat_pmf[good_to_eat][color] += 1
            self.softness_given_good_to_eat_pmf[good_to_eat][softness] += 1

        normalize_pmf(self.good_to_eat_prior)
    
        for good_to_eat in self.good_to_eat_prior:
            normalize_pmf(self.color_given_good_to_eat_pmf[good_to_eat])
            normalize_pmf(self.softness_given_good_to_eat_pmf[good_to_eat])
        
        return self


            
            
        
        
        
    def predict_color_proba(self: AvacadoPredictorType,
                            X: List[Color]
                            ) -> List[List[Tuple[GoodToEat, float]]]:
        
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()
        
        color_counts = defaultdict(float)
        
        for color in X:
            color_counts[color] += 1
        
        normalize_pmf(color_counts)
        
        for color in X:
            
            ListForProbs = []
            
            for good_to_eat in self.good_to_eat_prior:
                
               XGivenColor = (self.color_given_good_to_eat_pmf[good_to_eat][color] * self.good_to_eat_prior[good_to_eat]) / (color_counts[color])
               ListForProbs += [(good_to_eat , XGivenColor)]
               
            probs_per_example += [ListForProbs]

        return probs_per_example

    def predict_softness_proba(self: AvacadoPredictorType,
                               X: List[Softness]
                               ) -> List[List[Tuple[GoodToEat, float]]]:
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()

        softness_counts = defaultdict(float)
        
        for softness in X:
            softness_counts[softness] += 1
        
        normalize_pmf(softness_counts)
        
        for softness in X:
            
            ListForProbs = []
            
            for good_to_eat in self.good_to_eat_prior:
                
                XGivenSoftness = (self.softness_given_good_to_eat_pmf[good_to_eat][softness] * self.good_to_eat_prior[good_to_eat]) / (softness_counts[softness])
                ListForProbs += [(good_to_eat, XGivenSoftness)]
                
            probs_per_example += [ListForProbs]

        return probs_per_example


    # EXTRA CREDIT
    def predict_color(self: AvacadoPredictorType,
                      X: List[Color]
                      ) -> List[GoodToEat]:
        # TODO: complete me!
        
        data = self.predict_color_proba(X)
        
        listOfBest = []
        
        for [(val1, val12), (val2, val22)] in data:
            if val12 > val22:
                listOfBest.append(val1)
            else:
                listOfBest.append(val2)
            
        return listOfBest
            


    def predict_softness(self: AvacadoPredictorType,
                         X: List[Softness]
                         ) -> List[GoodToEat]:
        # TODO: complete me!
        
        data = self.predict_softness_proba(X)
        
        listOfBest = []
        
        for [(val1, val12), (val2, val22)] in data:
            if val12 > val22:
                listOfBest.append(val1)
            else:
                listOfBest.append(val2)
            
        return listOfBest
    
    




def accuracy(predictions: List[GoodToEat],
             actual: List[GoodToEat]
             ) -> float:
    if len(predictions) != len(actual):
        raise ValueError(f"ERROR: expected predictions and actual to be same length but got pred={len(predictions)}" +
            " and actual={len(actual)}")

    num_correct: float = 0
    for pred, act in zip(predictions, actual):
        num_correct += int(pred == act)

    return num_correct / len(predictions)


def main() -> None:
    data: List[Tuple[Color, Softness, GoodToEat]] = load_data()

    color_data: List[Color] = [color for color, _, _ in data]
    softness_data: List[Softness] = [softness for _, softness, _ in data]
    good_to_eat_data: List[GoodToEat] = [good_to_eat for _, _, good_to_eat in data]

    m: AvacadoPredictor = AvacadoPredictor().fit(data)

    print("good to eat prior")
    pprint(m.good_to_eat_prior)
    print()
    print()

    print("color given good to eat pmf")
    pprint(m.color_given_good_to_eat_pmf)
    print()
    print()

    print("softness given good to eat pmf")
    pprint(m.softness_given_good_to_eat_pmf)


    color_predictions = m.predict_color_proba(color_data)
    print(color_data)
    print("Predictions based on color:")
    pprint(color_predictions)
    # if you do the extra credit be sure to uncomment these lines!
    # print("accuracy when predicting only on color: ", accuracy(m.predict_color(color_data), good_to_eat_data))

    # print("accuracy when predicting only on softness: ", accuracy(m.predict_softness(softness_data), good_to_eat_data))


if __name__ == "__main__":
    main()

