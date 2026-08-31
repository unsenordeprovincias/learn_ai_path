from mnist.domain.models import Perceptron
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

