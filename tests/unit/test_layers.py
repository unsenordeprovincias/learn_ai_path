from mnist.domain.models import Vector, Perceptron, Layer
from mnist.domain.functions import relu

def test_create_layer():
    layer = Layer(784, 16, relu)
    assert isinstance(layer, Layer)
    assert len(layer) == 16
    assert len(layer.perceptrons) == 16
    p = layer.perceptrons[0]
    assert len(p.weights) == 784


def test_output_layer():
    layer = Layer(2, 2, relu)
    layer[0].weights = Vector([1, 1])
    layer[0].bias = -0.5
    layer[1].weights = Vector([1, 1])
    layer[1].bias = -1.5

    output = layer.output(Vector([0, 1]))
    assert isinstance(output, Vector)
    assert len(output) == 2
    assert output[0] == 0.5
    assert output[1] == 0