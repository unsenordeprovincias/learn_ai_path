# ports/raw_sample_sequence.py
from typing import Protocol

from mnist.domain.samples import RawSample


class RawSampleSequence(Protocol):
    """Secuencia de solo lectura de RawSample, sin estado de recorrido.

    "Raw" = sin transformar para el modelo: cada ejemplo son los bytes tal como
    estan guardados. Normalizar o pasar la etiqueta a one-hot es trabajo de quien
    consuma la secuencia, no de ella.

    Cada x[i] es un par (x, y_true), en ese orden. La convencion de que el registro
    i de las imagenes se empareja con el registro i de las etiquetas es de MNIST,
    no del formato IDX: un fichero IDX suelto no sabe con quien se empareja.

    No hay iterador ni posicion actual: leer x[i] no cambia lo que devolvera la
    siguiente lectura. Quien quiera recorrer o barajar elige los indices.
    """

    def __len__(self) -> int:
        """Numero de ejemplos n. Los indices validos son 0 <= i < n."""
        ...

    @property
    def x_shape(self) -> tuple[int, ...]:
        """Forma de la entrada de un ejemplo, p. ej. (28, 28).

        Es propiedad del dato, no del formato: el port no expone magic, numero de
        dimensiones ni cabecera. Los bytes de RawSample.x siguen el orden de IDX,
        donde el ultimo indice es el que varia mas rapido (para (28, 28): fila a fila).
        """
        ...

    def __getitem__(self, i: int) -> RawSample:
        """Devuelve el ejemplo i como RawSample(x, y_true).

        Lanza IndexError para todo i fuera de 0 <= i < n, negativos incluidos.
        Es una desviacion deliberada de la semantica de secuencia de Python, donde
        x[-1] es el ultimo: aqui un -1 casi siempre es un error de calculo de
        indices y preferimos que falle a que lea en silencio el ultimo ejemplo.
        """
        ...
