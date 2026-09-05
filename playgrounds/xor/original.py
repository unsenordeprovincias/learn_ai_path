from mnist.domain.models import Layer, Vector

_step = lambda x: 0 if x < 0 else 1

layer1 = Layer(2, 2, _step)
layer1[0].weights = Vector([1, 1])
layer1[0].bias = -0.5
layer1[1].weights = Vector([1, 1])
layer1[1].bias = -1.5

layer2 = Layer(2, 1, _step)
layer2[0].weights = Vector([1, -2])
layer2[0].bias = -0.5


def XOR(x1: int, x2: int) -> float:
    v = layer1.output(Vector([x1, x2]))
    r = layer2.output(v)
    return r[0]
