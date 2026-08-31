from mnist.domain.models import Perceptron, Vector
import random
import pytest

random.seed(5)
PARAMS = 7
init_values = tuple([random.uniform(-1, 1) for _ in range(PARAMS)])
init_bias = random.random()
f = lambda x: 0 if x < 0 else x

@pytest.fixture
def perceptron():
    random.seed(5)
    return Perceptron(PARAMS, f) 

def test_create_perceptron(perceptron):
    weights = perceptron.weights
    bias = perceptron.bias

    assert weights.values == init_values
    assert bias == init_bias
    assert perceptron.f_activation == f

def test_ponderated_sum(perceptron):

    input_signal = Vector([1] * PARAMS)
    weighted_sum = perceptron.weighted_sum(input_signal)

    assert weighted_sum == sum(init_values) + init_bias

def test_forward(perceptron):
    input_signal = Vector([1] * PARAMS)
    weighted_sum = perceptron.weighted_sum(input_signal)
    output_signal = 0 if weighted_sum < 0 else weighted_sum

    assert output_signal == perceptron.forward(input_signal)

def test_str_perceptron(perceptron):
    expected = (
        f"Perceptron(inputs={PARAMS}, "
        f"bias={perceptron.bias:.4f}, "
        f"activation={perceptron.f_activation.__name__})"
    )
    assert repr(perceptron) == expected

