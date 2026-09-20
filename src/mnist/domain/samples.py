from dataclasses import dataclass

from mnist.domain.models import Vector


@dataclass(frozen=True)
class RawSample:
    """Un par (x, y_true) tal como esta guardado, sin transformar para el modelo.

    "Raw" = sin transformar para el modelo: son los bytes del dato, no una
    entrada normalizada ni un one-hot. Ningun campo es un Vector.

    El orden del par es siempre (entrada, objetivo): x primero, y_true despues.
    """

    x: bytes
    y_true: bytes


@dataclass(frozen=True)
class NetSample:
    """Un par (x, y_true) ya transformado, listo para alimentar a la red.

    El orden del par es el mismo que en RawSample: x primero, y_true despues.
    """

    x: Vector
    y_true: Vector
