from collections.abc import Iterable, Callable
from random import uniform, random
from functools import reduce
from dataclasses import dataclass



class Vector:
    def __init__(self, values: Iterable[float]):
        for ix, item in enumerate(values):
            if not isinstance(item, (int, float)):
                raise TypeError(f"Item {ix}, value = {item}, type = '{type(item).__name__}' must be float")
        self.__values = tuple(values)

    @classmethod
    def __class_getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args, )

        return cls(args)  

    @classmethod
    def from_values(cls, *values):
        return cls(values)

    @classmethod
    def initialize(cls, length:int):
        values = [uniform(-1, 1) for _ in range(length)]
        return cls(values)

    @property
    def values(self):
        return self.__values

    def __is_correct_type(self, other, operand, *types):
        if not isinstance(other, types):
            raise TypeError(f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'")

    def __len__(self):
        return len(self.__values)

    def __getitem__(self, key: int):
        return self.__values[key]    

    def __eq__(self, value):
        return isinstance(value, type(self)) and self.values == value.values

    def __add__(self, other: "Vector"):
        self.__is_correct_type(other, "+", Vector)
        if len(self) != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"({len(self)},) ({len(other)},)"
        )
        return Vector(tuple(a + b for a, b in zip(self.values, other.values)))

    def __sub__(self, other: "Vector"):
        self.__is_correct_type(other, "-", Vector)
        if len(self) != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"({len(self)},) ({len(other)},)"
        )
        other =  other * -1
        return Vector(tuple(a + b for a, b in zip(self.values, other.values)))


    def __mul__(self, other: int | float):
        self.__is_correct_type(other, "*", int, float)
        return Vector(tuple(x * other for x in self.values))

    def __rmul__(self, other: int | float):
        return self.__mul__(other)

    def __matmul__(self, other: "Vector"):
        self.__is_correct_type(other, "@", Vector)
        if len(self) != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"({len(self)},) ({len(other)},)"
        )
        return sum(a * b for a, b in zip(self.values, other.values))
        
    def __repr__(self):
        return f"Vector {self.values}"

@dataclass
class Cache:
    input_signal: Vector
    output_signal: float
    weighted_sum: float

class Perceptron:
    def __init__(self, length: int, fActivation: Callable[[float], float]):
        self.weights = Vector.initialize(length)
        self.bias = random()
        self.f_activation = fActivation
        self.cache = None

    def weighted_sum(self, input_signal: Vector) -> float:
        return self.weights @ input_signal + self.bias

    def output(self, input_signal: Vector) -> float:
        weighted_sum = self.weighted_sum(input_signal) 
        output_signal = self.f_activation(weighted_sum)
        self.cache = Cache(input_signal, output_signal, weighted_sum)
        return output_signal

    def correct(self, delta_weights: Vector, delta_bias: float):
        self.weights += delta_weights
        self.bias += delta_bias

    def __repr__(self):
        return (
            f"Perceptron(inputs={len(self.weights)}, "
            f"bias={self.bias:.4f}, "
            f"activation={self.f_activation.__class__.__name__})"
        )

class Layer:
    def __init__(self, inputs: int, output: int, fActivation: Callable = None):
        f = (lambda x: 0 if x < 0 else x) if fActivation is None else fActivation

        self.perceptrons = [Perceptron(inputs, f) for _ in range(output)]

    def output(self, _input: Vector) -> Vector:
        result = [perceptron.output(_input) for perceptron in self.perceptrons]
        return Vector(result)

    def __len__(self):
        return len(self.perceptrons)

    def __getitem__(self, key: int):
        return self.perceptrons[key]

        
class NeuralNet:
    def __init__(self, layers: list[Layer], floss: Callable = None):
        self.__layers = layers
        self.__num_perceptrons = 0
        self.__floss = floss

    def __len__(self) -> int:
        return len(self.__layers)        

    def __getitem__(self, key: int) -> Layer:
        return self.__layers[key]

    @property
    def num_perceptrons(self):
        if not self.__num_perceptrons:
            self.__num_perceptrons = reduce(lambda accum, item: accum + len(item), self.__layers, 0)
        return self.__num_perceptrons

    @property
    def layers(self):
        return self.__layers

    @property
    def floss(self):
        return self.__floss

    def forward(self, input: Vector) -> Vector:
        for layer in self.layers:
            res = layer.output(input)
            input = res

        return res