from mnist.domain.models import Perceptron, Vector
import random

random.seed(5)
PARAMS = 7
init_values = tuple([random.uniform(-1, 1) for _ in range(PARAMS)])
init_bias = random.random()

def test_create_perceptron():
    random.seed(5)
    f = lambda x: 0 if x < 0 else x
    p = Perceptron(PARAMS, f)
    weights = p.weights
    bias = p.bias

    assert weights.values == init_values
    assert bias == init_bias
    assert p.f_activation == f

def test_ponderated_sum():
    random.seed(5)
    f = lambda x: 0 if x < 0 else x
    p = Perceptron(PARAMS, f)

    input_signal = Vector([1] * PARAMS)
    weighted_sum = p.weighted_sum(input_signal)

    assert weighted_sum == sum(init_values) + init_bias

