from mnist.domain.models import Perceptron, Vector
from mnist.domain.functions import relu
import random
import pytest

random.seed(5)
PARAMS = 7
init_values = tuple([random.uniform(-1, 1) for _ in range(PARAMS)])
init_bias = random.random()

@pytest.fixture
def perceptron():
    random.seed(5)
    return Perceptron(PARAMS, relu) 

def test_create_perceptron(perceptron):
    weights = perceptron.weights
    bias = perceptron.bias

    assert weights.values == init_values
    assert bias == init_bias
    assert perceptron.f_activation == relu

def test_ponderated_sum(perceptron):

    input_signal = Vector([1] * PARAMS)
    weighted_sum = perceptron.weighted_sum(input_signal)

    assert weighted_sum == sum(init_values) + init_bias

def test_output(perceptron):
    input_signal = Vector([1] * PARAMS)
    weighted_sum = perceptron.weighted_sum(input_signal)
    output_signal = 0 if weighted_sum < 0 else weighted_sum

    assert output_signal == perceptron.output(input_signal)
    assert perceptron.cache.input_signal == input_signal
    assert perceptron.cache.weighted_sum == weighted_sum
    assert perceptron.cache.output_signal == output_signal


def test_str_perceptron(perceptron):
    expected = (
        f"Perceptron(inputs={PARAMS}, "
        f"bias={perceptron.bias:.4f}, "
        f"activation={perceptron.f_activation.__class__.__name__})"
    )
    assert repr(perceptron) == expected

def test_correct_perceptron(perceptron):
    delta_weights = Vector([1, -1, 0, 0, 0, 0, 0])
    delta_bias = -0.5

    expected_weights = delta_weights + perceptron.weights
    expected_bias = delta_bias + perceptron.bias

    perceptron.correct(delta_weights, delta_bias)

    assert perceptron.bias == expected_bias
    assert perceptron.weights == expected_weights

