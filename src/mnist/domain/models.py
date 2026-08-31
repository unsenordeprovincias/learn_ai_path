from collections.abc import Iterable, Callable
from random import uniform, random

class Vector:
    def __init__(self, values: Iterable[float]):
        for ix, item in enumerate(values):
            if not isinstance(item, (int, float)):
                raise TypeError(f"Item {ix}, value = {item}, type = '{type(item).__name__}' must be float")
        self.__values = tuple(values)

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

class Perceptron:
    def __init__(self, length: int, fActivation: Callable[[float], float]):
        self.weights = Vector.initialize(length)
        self.bias = random()
        self.f_activation = fActivation

    def weighted_sum(self, input_signal: Vector) -> float:
        return self.weights @ input_signal + self.bias

    def output(self, input_signal: Vector) -> float:
        return self.f_activation(self.weighted_sum(input_signal))

    def correct(self, delta_weights: Vector, delta_bias: float):
        self.weights += delta_weights
        self.bias += delta_bias

    def __repr__(self):
        return (
            f"Perceptron(inputs={len(self.weights)}, "
            f"bias={self.bias:.4f}, "
            f"activation={self.f_activation.__name__})"
        )