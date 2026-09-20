import math
from contextlib import ExitStack
from pathlib import Path

from mnist.domain.samples import RawSample


def _line_offsets(file, path: Path, expected_columns: int) -> list[int]:
    """Byte donde empieza cada linea, comprobando de paso que todas tienen el mismo ancho.

    Las lineas de un CSV miden distinto ("0" ocupa 1 caracter, "255" ocupa 3), asi que a
    diferencia de IDX no hay una formula cabecera + i * tamano: la unica forma de saber
    donde empieza la linea i es haber recorrido las anteriores. Solo se guardan los
    desplazamientos, no el contenido.
    """
    offsets = []
    position = 0
    first_blank_line = None
    for line_number, line in enumerate(file, start=1):
        row = line.rstrip(b"\r\n")
        if not row:
            # en blanco al final del fichero se tolera; solo es error si despues hay datos
            first_blank_line = first_blank_line or line_number
            position += len(line)
            continue
        if first_blank_line:
            raise ValueError(f"{path}: blank line {first_blank_line} in the middle of the file")
        columns = row.count(b",") + 1
        if columns != expected_columns:
            raise ValueError(
                f"{path}: line {line_number} has {columns} columns but x_shape implies "
                f"{expected_columns} (1 label + the pixels of one example)"
            )
        offsets.append(position)
        position += len(line)
    if not offsets:
        raise ValueError(f"{path}: the file has no samples")
    return offsets


class CsvRawSampleSequence:
    """RawSampleSequence sobre un CSV de MNIST: una fila por ejemplo, la etiqueta en la
    primera columna, despues los pixeles como enteros, sin cabecera.

    x_shape lo da quien construye, porque el fichero no la lleva: una fila es una lista
    plana de numeros y (28, 28) o (784,) serian igual de validos. Se comprueba que
    1 + producto(x_shape) coincide con el numero de columnas de cada fila.

    Al abrir se recorre el fichero una vez para guardar donde empieza cada linea y para
    validar su ancho, las lineas en blanco y que haya al menos un ejemplo. El contenido de
    las celdas NO se valida al abrir sino al leer x[i], que lanza ValueError si una celda
    esta vacia, no es un entero o queda fuera de 0-255 (bytes no puede guardar mas). Un
    pixel corrupto se descubre por tanto cuando alguien lee esa fila, no antes.

    Limitaciones conocidas: int() acepta " 5", "+5" y "1_0" (= 10), asi que esas celdas
    pasan; un CSV con pixeles decimales ("255.0") se rechaza. Si el fichero cambia despues
    de abrirlo, los desplazamientos guardados dejan de valer.
    """

    def __init__(self, path: Path, x_shape: tuple[int, ...]):
        with ExitStack() as stack:
            file = stack.enter_context(open(path, "rb"))
            self._offsets = _line_offsets(file, path, 1 + math.prod(x_shape))
            self._file = file
            self._x_shape = x_shape
            self._stack = stack.pop_all()

    def __enter__(self) -> "CsvRawSampleSequence":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stack.close()

    def __len__(self) -> int:
        return len(self._offsets)

    @property
    def x_shape(self) -> tuple[int, ...]:
        return self._x_shape

    def __getitem__(self, i: int) -> RawSample:
        # una lista aceptaria los negativos (offsets[-1] es el ultimo); el port no
        if not 0 <= i < len(self._offsets):
            raise IndexError(f"sample index {i} is out of range for {len(self._offsets)} samples")
        self._file.seek(self._offsets[i])
        label, *pixels = self._file.readline().rstrip(b"\r\n").split(b",")
        return RawSample(x=bytes(int(pixel) for pixel in pixels), y_true=bytes([int(label)]))
