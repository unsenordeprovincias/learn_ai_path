# tests/unit/test_backward_jacobian_convention.py
from mnist.domain.models import Vector, Matrix, Layer, NeuralNet
from mnist.domain.functions import Activation, Loss


class FakeAsymmetricActivation(Activation):
    """Solo para este test: Jacobiano constante y NO simétrico
    (A[0][1]=3 != A[1][0]=1), para que J@g y J.T@g difieran."""

    A = Matrix([[2, 3], [1, 5]])

    def __call__(self, z: Vector) -> Vector:
        return self.A @ z

    def derivative(self, z: Vector) -> Matrix:
        return self.A


class PickFirstOutput(Loss):
    """L = a_out[0]. Fuerza g = [1, 0], el caso más simple de leer."""

    def __call__(self, a_out: Vector, y_true: Vector) -> float:
        return a_out[0]

    def derivative(self, a_out: Vector, y_true: Vector) -> Vector:
        return Vector([1.0, 0.0])


def test_backward_usa_jacobiano_transpuesto_no_directo():
    layer = Layer(inputs=2, output=2, fActivation=FakeAsymmetricActivation())
    net = NeuralNet(layers=[layer], floss=PickFirstOutput())

    net.forward(Vector([1.0, 1.0]))  # los pesos reales no importan: ver razonamiento arriba
    net.backward(y_true=Vector([0.0, 0.0]), learning_rate=0.0)

    # A.T @ [1,0] = [2, 3] -- correcto
    # A   @ [1,0] = [2, 1] -- lo que hace hoy backward()
    assert layer[0].cached_gradient.bias == 2.0  # coincide en ambas convenciones, no discrimina
    assert layer[1].cached_gradient.bias == 3.0  # ESTE es el que falla hoy: da 1.0, no 3.0