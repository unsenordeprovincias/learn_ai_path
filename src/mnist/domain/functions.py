from abc import ABC, abstractmethod
from math import exp
from mnist.domain.models import Vector, Matrix
from typing import Callable

class Differentiable(ABC):
    """Base: toda función con derivada"""
    
    @abstractmethod
    def __call__(self, *args):
        pass
    
    @abstractmethod
    def derivative(self, *args):
        pass


class Activation(Differentiable):
    @abstractmethod
    def __call__(self, z: Vector) -> Vector:
        """Aplica la función de activación"""
        pass

    @abstractmethod
    def derivative(self, z: Vector) -> Matrix:
        """Derivada analítica"""
        pass

    def __repr__(self):
        return f"fActivation -> {self.__class__.__name__}"


class IndependentActivation(Activation):
    # El nombre deberia ser PointwiseActivation, pero esta noche tiene mas sentido Independent
    """Toda activación independiente comparte esta forma:
    aplica una función escalar componente a componente,
    y su derivada es siempre una matriz diagonal."""
    def __init__(self, fn: Callable[[float], float], fn_derivative: Callable[[float], float]):
        self._fn = fn
        self._fn_derivative = fn_derivative

    def __call__(self, z: Vector) -> Vector:
        return Vector([self._fn(x) for x in z.values])

    def derivative(self, z: Vector) -> Matrix:
        n = len(z)
        return Matrix([
            [self._fn_derivative(z[i]) if i == j else 0.0 for j in range(n)]
            for i in range(n)
        ])


sigmoid_fn = lambda x: 1 / (1 + exp(-x))
sigmoid_derivative = lambda x: sigmoid_fn(x) * (1 - sigmoid_fn(x))
sigmoid = IndependentActivation(
    fn=sigmoid_fn,
    fn_derivative=sigmoid_derivative
)

relu_fn = lambda x: 0 if x < 0 else x
relu_derivative = lambda x: 0 if x < 0 else 1
relu = IndependentActivation(
    fn=relu_fn,
    fn_derivative=relu_derivative
)


class Loss(Differentiable):
    """Subclase abstracta: (a_out, y_true) → L"""
    
    @abstractmethod
    def __call__(self, a_out, y_true):
        pass
    
    @abstractmethod
    def derivative(self, a_out, y_true):
        pass

    def __repr__(self):
        return f"fLoss -> {self.__class__.__name__}"


class MSE(Loss):
    def __call__(self, a_out: Vector, y_true: Vector):
        return 0.5 * ((a_out - y_true) @ (a_out - y_true))
    
    def derivative(self, a_out, y_true):
        return a_out - y_true

mse = MSE()