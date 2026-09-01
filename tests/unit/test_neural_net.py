from mnist.domain.models import Layer, NeuralNet, Vector

relu = lambda x: 0 if x < 0 else x


def test_create_neural_net():
    l1 = Layer(2, 2, relu)
    loutput = Layer(2, 1, relu)
    nnXOR = NeuralNet([l1, loutput])

    assert isinstance(nnXOR, NeuralNet)
    assert len(nnXOR) == 2
    assert nnXOR[0] == l1
    assert nnXOR[1] == loutput

    assert nnXOR.num_perceptrons == 3

def test_forward_neural_net():
    _step = lambda x: 0 if x < 0 else 1

    l1 = Layer(2, 2, _step)
    l1[0].weights = Vector([1, 1])
    l1[0].bias = -0.5
    l1[1].weights = Vector([1, 1])
    l1[1].bias = -1.5

    loutput = Layer(2, 1, _step)
    loutput[0].weights = Vector([1, -2])
    loutput[0].bias = -0.5

    nnXOR = NeuralNet([l1, loutput])

    assert nnXOR.forward(Vector([0, 0])) == Vector([0])
    assert nnXOR.forward(Vector([0, 1])) == Vector([1])
    assert nnXOR.forward(Vector([1, 0])) == Vector([1])
    assert nnXOR.forward(Vector([1, 1])) == Vector([0])

    