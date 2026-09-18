from mnist.domain.functions import relu, sigmoid, Activation
from mnist.domain.models import Vector, Matrix
import pytest

def test_activation():
    f = relu

    assert isinstance(f, Activation)
    assert f(Vector[-1]) == Vector[0]
    assert f(Vector[0.5]) == Vector[0.5]
    assert f.derivative(Vector[-1]) == Matrix([[0]])
    assert f.derivative(Vector[0.5]) == Matrix([[1]])

    g = sigmoid

    assert isinstance(g, Activation)
    der = g.derivative(Vector[-1]) 
    assert der[0][0] == pytest.approx(0.19661193, abs=1e-6)

    der = g.derivative(Vector[0.5]) 
    assert der[0][0] == pytest.approx(0.23500371, abs=1e-6)
