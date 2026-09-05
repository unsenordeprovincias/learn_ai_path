from mnist.domain.models import Layer, NeuralNet, Vector, Cache
from mnist.domain.functions import relu, mse, sigmoid
import pytest
from tests.conftest import numerical_gradient

cache_l1_0 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + 1*1 + (-0.5),  # = 1.5
    output_signal=1,            # = 1
    weights = Vector[1,1]
)

# l1[1]: pesos [1, 1], bias -1.5
cache_l1_1 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + 1*1 + (-1.5),  # = 0.5
    output_signal=1,            # = 1
    weights = Vector[1,1]
)

cache_l1 = Cache(
    input_signal = cache_l1_0.input_signal,
    weighted_sum = Vector[cache_l1_0.weighted_sum, cache_l1_1.weighted_sum],
    output_signal= Vector[cache_l1_0.output_signal, cache_l1_1.output_signal],
    weights=None
)

# loutput[0]: pesos [1, -2], bias -0.5
# Entrada: [cache_l1_0.output_signal, cache_l1_1.output_signal] = [1, 1]
cache_loutput_0 = Cache(
    input_signal=Vector[1, 1],
    weighted_sum=1*1 + (-2)*1 + (-0.5),  # = -1.5
    output_signal=0,              # = 0
    weights= Vector[1, -2]
)

cache_loutput = Cache(
    input_signal = cache_loutput_0.input_signal,
    weighted_sum = Vector[cache_loutput_0.weighted_sum],
    output_signal = Vector[cache_loutput_0.output_signal],
    weights=None
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

    expected_p_caches = [cache_l1_0, cache_l1_1, cache_loutput_0]
    expected_l_caches = [cache_l1, cache_loutput]

    l_ix = p_ix = 0

    for layer in nnXOR.layers:
        assert layer.cache.output_signal == expected_l_caches[l_ix].output_signal
        assert layer.cache.input_signal == expected_l_caches[l_ix].input_signal
        assert layer.cache.weighted_sum == expected_l_caches[l_ix].weighted_sum
        for perceptron in layer:
            assert perceptron.cache.input_signal == expected_p_caches[p_ix].input_signal
            assert perceptron.cache.weighted_sum == expected_p_caches[p_ix].weighted_sum
            assert perceptron.cache.output_signal == expected_p_caches[p_ix].output_signal
            p_ix += 1
        l_ix += 1

@pytest.mark.parametrize("f_activacion",
                         [relu, sigmoid])
def test_backward_gradient_cached(f_activacion):                   

    
    l1 = Layer(2, 2, f_activacion)
    loutput = Layer(2, 1, f_activacion)
    nnAny = NeuralNet([l1, loutput], mse)

    x = Vector[1, 0]
    y_true = Vector[1]

    output = nnAny.forward(x)
    nnAny.backward(y_true, learning_rate=0)

    grads = numerical_gradient(nnAny, x, y_true)

    for (layer_idx, neuron_idx), grad_num in grads.items():
        grad_ana = nnAny.layers[layer_idx][neuron_idx].cached_gradient

        for w_ana, w_num in zip(grad_ana.weights.values, grad_num.weights.values):
            assert w_ana == pytest.approx(w_num, rel=1e-4)
        assert grad_ana.bias == pytest.approx(grad_num.bias, rel=1e-4)

        print(f"✓ Layer {layer_idx}, Neuron {neuron_idx}: {grad_ana} ≈ {grad_num}")