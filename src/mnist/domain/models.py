from collections.abc import Iterable, Callable
from random import uniform, random
from functools import reduce
from dataclasses import dataclass

class Vector:
    def __init__(self, values: Iterable[float]):
        for ix, item in enumerate(values):
            if not isinstance(item, (int, float)):
                raise TypeError(f"Item {ix}, value = {item}, type = '{type(item).__name__}' must be float")
        self.__values = tuple(values)

    # mypy lee Vector[...] como una aplicacion de tipo generico y lo rechaza (una metaclase
    # con __getitem__ tampoco lo arregla), asi que cada uso lleva `# type: ignore[misc]`.
    @classmethod
    def __class_getitem__(cls, args):
        if not isinstance(args, tuple):
            args = (args, )

        return cls(args)  

    @classmethod
    def from_values(cls, *values):
        return cls(values)

    @classmethod
    def initialize(cls, length:int):
        values = [uniform(-1, 1) for _ in range(length)]
        return cls(values)

    @classmethod
    def one_hot(cls, pos: int, length: int):
        if pos < 0 or pos >= length:
            raise ValueError(f"one_hot position {pos} is out of range for length {length}")
        return cls([1.0 if i == pos else 0.0 for i in range(length)])
    
    @property
    def values(self):
        return self.__values

    def __is_correct_type(self, other, operand, *types):
        if not isinstance(other, types):
            raise TypeError(f"unsupported operand type(s) for {operand}: '{type(self).__name__}' and '{type(other).__name__}'")

    def __are_same_length(self, other):
        if len(self) != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"({len(self)},) ({len(other)},)"
        )

    def __len__(self):
        return len(self.__values)

    def __getitem__(self, key: int):
        return self.__values[key]    

    def __eq__(self, value):
        return isinstance(value, type(self)) and self.values == value.values

    def __add__(self, other: "Vector"):
        self.__is_correct_type(other, "+", Vector)
        self.__are_same_length(other)
        return Vector(tuple(a + b for a, b in zip(self.values, other.values)))

    def __sub__(self, other: "Vector"):
        self.__is_correct_type(other, "-", Vector)
        self.__are_same_length(other)
        other =  other * -1
        return Vector(tuple(a + b for a, b in zip(self.values, other.values)))


    def __mul__(self, other: "int | float | Vector"):
        self.__is_correct_type(other, "*", int, float, Vector)

        if isinstance(other, Vector):
            self.__are_same_length(other)
            return Vector(tuple(a * b for a, b in zip(self.values, other.values)))

        return Vector(tuple(x * other for x in self.values))

    def __rmul__(self, other: int | float):
        return self.__mul__(other)

    def __matmul__(self, other: "Vector"):
        if isinstance(other, Matrix):
            return NotImplemented  # deja que Python intente Vector.__rmatmul__
        
        self.__is_correct_type(other, "@", Vector)
        if len(self) != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"({len(self)},) ({len(other)},)"
        )
        return sum(a * b for a, b in zip(self.values, other.values))
        
    def __repr__(self):
        return f"Vector {self.values}"

class Matrix:
    def __init__(self, rows: Iterable[Vector | Iterable[float]]):
        vectors = tuple(row if isinstance(row, Vector) else Vector(row) for row in rows)
        if vectors and any(len(row) != len(vectors[0]) for row in vectors):
            raise ValueError(
                f"all rows must have the same length, got lengths "
                f"{tuple(len(row) for row in vectors)}"
            )
        self.__rows = vectors

    @property
    def rows(self):
        return self.__rows

    @property
    def shape(self):
        if not self.__rows:
            return (0, 0)
        return (len(self.__rows), len(self.__rows[0]))

    @property
    def T(self):
        columns = zip(*(row.values for row in self.__rows))
        return Matrix([list(column) for column in columns])

    def __matmul__(self, other: "Vector"):
        if not isinstance(other, Vector):
            raise TypeError(f"unsupported operand type(s) for @: 'Matrix' and '{type(other).__name__}'")
        if self.shape[1] != len(other):
            raise ValueError(
                f"operands could not be broadcast together with shapes "
                f"{self.shape} ({len(other)},)"
            )
        return Vector([row @ other for row in self.__rows])

    def __rmatmul__(self, other: "Vector") -> "Vector":
        # se invoca cuando: other(Vector) @ self(Matrix)
        # y Vector.__matmul__ devolvió NotImplemented
        return self.T @ other

    def __getitem__(self, key):
        return self.__rows[key]

    def __repr__(self):
        return f"Matrix {tuple(row.values for row in self.__rows)}"

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.__rows == other.rows

    def __hash__(self):
        return hash(self.__rows)

@dataclass
class NetSample:
    x: Vector
    y_true: Vector

@dataclass
class PerceptronCache:
    input_signal: Vector
    weighted_sum: float
    weights: Vector

@dataclass
class LayerCache:
    input_signal: Vector
    weighted_sum: Vector     # el vector reunido de todos los perceptrones
    output_signal: Vector    # ya activado, resultado de f_activation sobre weighted_sum

@dataclass
class CachedGradient:
    weights: Vector
    bias: float

class Perceptron:
    def __init__(self, inputs_length: int):
        self.weights = Vector.initialize(inputs_length)
        self.bias = random()
        self.cache: PerceptronCache | None = None
        self.cached_gradient: CachedGradient | None = None

    def weighted_sum(self, input_signal: Vector) -> float:
        return self.weights @ input_signal + self.bias

    def output(self, input_signal: Vector) -> float:
        weighted_sum = self.weighted_sum(input_signal) 
        self.cache = PerceptronCache(
            input_signal = input_signal, 
            weighted_sum = weighted_sum,
            weights = Vector(self.weights.values)
        )
        return weighted_sum

    def correct(self, learning_rate: float):
        if not learning_rate:
            return
        assert self.cached_gradient is not None, "backward must compute the gradient before correct"
        self.weights += (-learning_rate * self.cached_gradient.weights)
        self.bias += (-learning_rate * self.cached_gradient.bias)

    def __repr__(self):
        return f"Perceptron(inputs={len(self.weights)}, bias={self.bias:.4f})"

class Layer:
    def __init__(self, inputs: int, output: int, fActivation: Callable):
        self.__f_activation = fActivation

        self.perceptrons = [Perceptron(inputs) for _ in range(output)]
        self.cache: LayerCache | None = None


    def output(self, _input: Vector) -> Vector:
        weighted_sums = Vector([perceptron.output(_input) for perceptron in self.perceptrons])
        self.cache = LayerCache(
            input_signal = _input,
            weighted_sum=weighted_sums,
            output_signal=self.f_activation(weighted_sums),
        )

        return self.cache.output_signal

    def __len__(self):
        return len(self.perceptrons)

    def __getitem__(self, key: int):
        return self.perceptrons[key]

    @property
    def f_activation(self):
        return self.__f_activation

    def __repr__(self):
        inputs = len(self.perceptrons[0].weights) if self.perceptrons else 0
        return (
            f"Layer(inputs={inputs}, outputs={len(self.perceptrons)}, "
            f"activation={self.__f_activation.__class__.__name__})"
        )
        
class NeuralNet:
    def __init__(self, layers: list[Layer], floss: Callable | None = None):
        self.__layers = layers
        self.__num_perceptrons = 0
        self.__floss = floss
        self.__last_output = None

    def __len__(self) -> int:
        return len(self.__layers)        

    def __getitem__(self, key: int) -> Layer:
        return self.__layers[key]

    @property
    def num_perceptrons(self):
        if not self.__num_perceptrons:
            self.__num_perceptrons = reduce(lambda accum, item: accum + len(item), self.__layers, 0)
        return self.__num_perceptrons

    @property
    def layers(self):
        return self.__layers

    @property
    def floss(self):
        return self.__floss

    @property
    def last_output(self):
        return self.__last_output

    def forward(self, input: Vector) -> Vector:
        for layer in self.layers:
            res = layer.output(input)
            input = res

        self.__last_output = res

        return res
    

    def backward(self, y_true: Vector, learning_rate: float = 0.0):
        """
        Propaga error hacia atrás y calcula gradientes.
        
        Args:
            y_true: Vector de salida esperada
            learning_rate: factor de aprendizaje (si es 0, solo diagnostica)
        """

        output = self.last_output
        loss_grad = self.floss.derivative(output, y_true)

        for ix_layer, layer in enumerate(reversed(self.layers)):
            '''
            activation_sensivities = Vector(tuple(map(layer.f_activation.derivative, layer.cache.weighted_sum)))
            local_error = loss_grad @ activation_sensivities
            '''
            jacobian = layer.f_activation.derivative(layer.cache.weighted_sum)
            # Con J[i][j] = da_i/dz_j, la regla de la cadena da dL/dz_j = sum_i g_i * J[i][j],
            # es decir J^T g, que se escribe g @ J. El orden importa: solo con un Jacobiano
            # simetrico (diagonal, softmax) J g y J^T g coinciden.
            delta = loss_grad @ jacobian

            for pos_in_input, perceptron in enumerate(layer):
                local_error = delta[pos_in_input]
                # diagnosis y grabacion
                weight_grad = local_error * layer.cache.input_signal 
                bias_grad = local_error

                perceptron.cached_gradient = CachedGradient(
                    weights = weight_grad,
                    bias = bias_grad
                )

                # corregir
                perceptron.correct(learning_rate)

            # Propaga loss_grad segun la regla de la cadena a siguiente capa
            if ix_layer < len(self.layers) - 1:
                W_transposed = Matrix([perceptron.cache.weights for perceptron in layer]).T
                loss_grad = W_transposed @ delta
                

