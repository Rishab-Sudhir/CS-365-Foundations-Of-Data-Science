# SYSTEM IMPORTS
from enum import Enum
from typing import List, Tuple, Type
import csv
import os


# PYTHON PROJECT IMPORTS



# types defined in this module
ColorType: Type = Type["Color"]
SoftnessType: Type = Type["Softness"]
GoodToEatType: Type = Type["GoodToEat"]


class Color(Enum):
    BLACK = 0
    BROWN = 1
    GREEN = 2

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")



class Softness(Enum):
    MUSHY = 0
    SOFT = 1
    TENDER = 2
    HARD = 3

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")


class GoodToEat(Enum):
    YES = 0
    NO = 1

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")


def load_data() -> List[Tuple[Color, Softness, GoodToEat]]:
    cd: str = os.path.dirname(__file__)
    data_dir: str = os.path.join(cd, "..", "data")
    if not os.path.exists(data_dir):
        raise Exception(f"ERROR: data_dir {data_dir} does not exist!")

    data_file: str = os.path.join(data_dir, "train_avacados.txt")
    if not os.path.exists(data_file):
        raise Exception(f"ERROR: file {data_file} does not exist!")

    data: List[Tuple[Color, Softness, GoodToEat]] = list()
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",")

        for line in reader:
            if len(line) > 0:

                if len(line) != 3:
                    raise ValueError(f"ERROR: expected three values but got {line}")

                color, softness, good_to_eat = line
                data.append(tuple([Color.value_of(color.strip().rstrip()),
                                   Softness.value_of(softness.strip().rstrip()),
                                   GoodToEat.value_of(good_to_eat.strip().rstrip())]))

    return data

