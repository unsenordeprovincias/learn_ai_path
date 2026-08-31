from mnist.domain.models import Vector, Perceptron, Layer

def test_create_layer():
    layer = Layer(784, 16)
    assert isinstance(layer, Layer)
    assert len(layer) == 16
    assert len(layer.perceptrons) == 16
    p = layer.perceptrons[0]
    assert len(p.weights) == 784