from abc import ABC, abstractmethod
from math import exp
from mnist.domain.models import Vector

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
    def __call__(self, z):
        """Aplica la función de activación"""
        pass

    @abstractmethod
    def derivative(self, z):
        """Derivada analítica"""
        pass

    def __repr__(self):
        return f"fActivation -> {self.__class__.__name__}"


class sigmoid(Activation):
    def __call__(self, z):
        return 1 / (1 + exp(-z))

    def derivative(self, z):
        a = self(z)
        return a * (1 - a)

sigmoid = sigmoid()
    
class relu(Activation):
    def __call__(self, z):
        return 0 if z < 0 else z

    def derivative(self, z):
        return 0 if z < 0 else 1

relu = relu()

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