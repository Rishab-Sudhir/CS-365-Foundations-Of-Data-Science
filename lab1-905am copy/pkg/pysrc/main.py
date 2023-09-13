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


class AvacadoPredictor(object):
    def __init__(self: AvacadoPredictorType) -> None:
        self.color_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Color, float]] = defaultdict(lambda: defaultdict(float))
        self.softness_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Softness, float]] = defaultdict(lambda: defaultdict(float))
        self.good_to_eat_prior: Dict[GoodToEat, float] = defaultdict(float)


    def fit(self: AvacadoPredictorType,
            data: List[Tuple[Color, Softness, GoodToEat]]
            ) -> AvacadoPredictorType: 
        
        goodToEat_count = sum(1 for _, _, good in data if good == GoodToEat.YES)
        
        TotalItems = len(data)
        
        self.good_to_eat_prior[GoodToEat.YES] = goodToEat_count/TotalItems
        self.good_to_eat_prior[GoodToEat.NO] =  1 - (goodToEat_count/TotalItems)
        
        color_good_to_eat_counts = {color: 0 for color in Color}
        
        for color, _, good_to_eat in data:
            if good_to_eat == GoodToEat.YES:
                    color_good_to_eat_counts[color] += 1
        
        for x in Color:
            self.color_given_good_to_eat_pmf[GoodToEat.YES][x] = color_good_to_eat_counts[x] / goodToEat_count
            self.color_given_good_to_eat_pmf[GoodToEat.NO][x] = 1 - ( color_good_to_eat_counts[x] / goodToEat_count)
          
        softness_good_to_eat_counts = {softness: 0 for softness in Softness}
            
        for _, softness, good_to_eat in data:
            if good_to_eat == GoodToEat.YES:
                    softness_good_to_eat_counts[softness] += 1
            
            for x in Softness:
                self.softness_given_good_to_eat_pmf[GoodToEat.YES][x] = softness_good_to_eat_counts[x] / goodToEat_count
                self.softness_given_good_to_eat_pmf[GoodToEat.NO][x] = 1 - ( softness_good_to_eat_counts[x] / goodToEat_count)
                
        # TODO: complete me!

        return self

    def predict_color_proba(self: AvacadoPredictorType,
                            X: List[Color]
                            ) -> List[List[Tuple[GoodToEat, float]]]:
        
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()
        
        color_counts = {color: 0 for color in Color}
        
        for color in X:
            color_counts[color] += 1
        
        TotalItems = len(X)
        
        for color in X:
            GoodGivenColor = (self.color_given_good_to_eat_pmf[GoodToEat.YES][color] * self.good_to_eat_prior[GoodToEat.YES]) / (color_counts[color] / TotalItems)
            probs_per_example += [[(GoodToEat.YES, GoodGivenColor), (GoodToEat.NO,1- GoodGivenColor)]]

        return probs_per_example

    def predict_softness_proba(self: AvacadoPredictorType,
                               X: List[Softness]
                               ) -> List[List[Tuple[GoodToEat, float]]]:
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()

        softness_counts = {softness: 0 for softness in Softness}
        
        for softness in X:
            softness_counts[softness] += 1
        
        TotalItems = len(X)
        
        for softness in X:
            GoodGivenSoftness = (self.softness_given_good_to_eat_pmf[GoodToEat.YES][softness] * self.good_to_eat_prior) / (softness_counts[softness] / TotalItems)
            probs_per_example += [[(GoodToEat.YES, GoodGivenSoftness), (GoodToEat.NO,1- GoodGivenSoftness)]]

        return probs_per_example


    # EXTRA CREDIT
    def predict_color(self: AvacadoPredictorType,
                      X: List[Color]
                      ) -> List[GoodToEat]:
        # TODO: complete me!
        return list()

    def predict_softness(self: AvacadoPredictorType,
                         X: List[Softness]
                         ) -> List[GoodToEat]:
        # TODO: complete me!
        return list()




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

