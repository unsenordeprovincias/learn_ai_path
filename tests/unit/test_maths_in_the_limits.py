# tests/unit/test_backward_jacobian_convention.py
from mnist.domain.models import Vector, Matrix, Layer, NeuralNet
from mnist.domain.functions import Activation, Loss, softmax, softmax_cross_entropy
from math import log



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

    net.forward(Vector([1.0, 1.0]))  # da igual con que pesos: el Jacobiano A y g son constantes
    net.backward(y_true=Vector([0.0, 0.0]), learning_rate=0.0)

    # g = [1, 0]; el bias_grad de cada perceptron es su componente de delta
    # J^T g = A.T @ g = [2, 3] -- convencion correcta
    # J   g = A   @ g = [2, 1] -- convencion incorrecta
    assert layer[0].cached_gradient.bias == 2.0  # igual en ambas convenciones: no discrimina
    assert layer[1].cached_gradient.bias == 3.0  # discrimina: J g daria 1.0, no 3.0


def test_softmax_cross_entropy_es_finita_con_diferencia_grande_de_logits():
    """Con diferencias de logit extremas (>700), softmax_cross_entropy
    debe devolver un número finito y positivo — nunca lanzar, nunca NaN."""
    z = Vector([0.0, -746.0])
    y = Vector([0.0, 1.0])  # clase correcta es la de logit muy bajo

    resultado = softmax_cross_entropy(z, y)

    assert resultado > 0
    assert resultado != float('inf')
    assert resultado == resultado  # descarta NaN


def test_softmax_cross_entropy_coincide_con_calculo_directo_en_caso_normal():
    """Control de correctitud: sin underflow, debe dar el mismo resultado
    que -log(p[c]) calculado directamente vía softmax."""
    z = Vector([2.0, 1.0, 0.5])
    y = Vector([0.0, 1.0, 0.0])

    resultado = softmax_cross_entropy(z, y)
    p = softmax(z)
    esperado = -log(p[1])

    assert abs(resultado - esperado) < 1e-9