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


    