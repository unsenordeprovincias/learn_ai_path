from mnist.domain.functions import relu, sigmoid, Activation
import pytest

def test_activation():
    f = relu

    assert isinstance(f, Activation)
    assert f(-1) == 0
    assert f(0.5) == 0.5
    assert f.derivative(-1) == 0
    assert f.derivative(0.5) == 1

    g = sigmoid

    assert isinstance(g, Activation)
    assert g(-1) == pytest.approx(0.26894142, abs=1e-6)
    assert g(0.5) == pytest.approx(0.62245933, abs=1e-6)
    assert g.derivative(-1) == pytest.approx(0.19661193, abs=1e-6)
    assert g.derivative(0.5) == pytest.approx(0.23500371, abs=1e-6)
