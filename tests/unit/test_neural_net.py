from mnist.domain.models import Layer, NeuralNet, Vector, Cache
from mnist.domain.functions import relu
import pytest

cache_l1_0 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + 1*1 + (-0.5),  # = 1.5
    output_signal=1            # = 1
)

# l1[1]: pesos [1, 1], bias -1.5
cache_l1_1 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + 1*1 + (-1.5),  # = 0.5
    output_signal=1            # = 1
)

# loutput[0]: pesos [1, -2], bias -0.5
# Entrada: [cache_l1_0.output_signal, cache_l1_1.output_signal] = [1, 1]
cache_loutput_0 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + (-2)*1 + (-0.5),  # = -1.5
    output_signal=0              # = 0
)

@pytest.fixture
def layers_XOR():
    _step = lambda x: 0 if x < 0 else 1

    l1 = Layer(2, 2, _step)
    l1[0].weights = Vector([1, 1])
    l1[0].bias = -0.5
    l1[1].weights = Vector([1, 1])
    l1[1].bias = -1.5

    loutput = Layer(2, 1, _step)
    loutput[0].weights = Vector([1, -2])
    loutput[0].bias = -0.5

    return l1, loutput


def test_create_neural_net():
    l1 = Layer(2, 2, relu)
    loutput = Layer(2, 1, relu)
    nnAny = NeuralNet([l1, loutput])

    assert isinstance(nnAny, NeuralNet)
    assert len(nnAny) == 2
    assert nnAny[0] == l1
    assert nnAny[1] == loutput

    assert nnAny.num_perceptrons == 3

def test_forward_neural_net(layers_XOR):
    nnXOR = NeuralNet(layers_XOR)

    assert nnXOR.forward(Vector([0, 0])) == Vector([0])
    assert nnXOR.forward(Vector([0, 1])) == Vector([1])
    assert nnXOR.forward(Vector([1, 0])) == Vector([1])
    assert nnXOR.forward(Vector([1, 1])) == Vector([0])

def test_cache_foward(layers_XOR):
    nnXOR = NeuralNet(layers_XOR)
    nnXOR.forward(Vector([1, 1]))

    expected_caches = [cache_l1_0, cache_l1_1, cache_loutput_0]
    ix = 0
    for layer in nnXOR.layers:
        for perceptron in layer:
            assert perceptron.cache.input_signal == expected_caches[ix].input_signal
            assert perceptron.cache.weighted_sum == expected_caches[ix].weighted_sum
            assert perceptron.cache.output_signal == expected_caches[ix].output_signal
            ix += 1
                                    
    