from mnist.domain.models import Layer, Vector

class MetaXOR(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls.layer1 = Layer(2, 2, lambda x: 0 if x < 0 else 1)
        cls.layer1[0].weights = Vector([1, 1])
        cls.layer1[0].bias = -0.5
        cls.layer1[1].weights = Vector([1, 1])
        cls.layer1[1].bias = -1.5

        cls.layer2 = Layer(2, 1, lambda x: 0 if x < 0 else 1)
        cls.layer2[0].weights = Vector([1, -2])
        cls.layer2[0].bias = -0.5


    def __call__(cls, x1: int, x2:int) -> float:
        v = cls.layer1.output(Vector([x1, x2]))
        r = cls.layer2.output(v)
        return r[0]
    
class XOR(metaclass=MetaXOR):
    pass
    