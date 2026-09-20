import math
import os
import struct
from contextlib import ExitStack
from pathlib import Path

from mnist.domain.samples import RawSample

_IMAGES_NDIMS = 3  # el ndims esperado viene de fuera del fichero: 3 para imagenes
_LABELS_NDIMS = 1  # y 1 para etiquetas


# Punto de extension: tipo IDX (tercer byte del magic) -> bytes por elemento.
# Hoy solo unsigned byte (0x08), el que usa MNIST. Anadir otro tipo aqui no basta: x[i]
# devuelve bytes sin decodificar, asi que habria que decidir tambien como se interpretan.
_ELEMENT_SIZE_BY_TYPE: dict[int, int] = {0x08: 1}


def _read_header_part(file, size: int, path: Path) -> bytes:
    part = file.read(size)
    if len(part) != size:
        raise ValueError(
            f"{path}: truncated header, needed {size} more bytes but only {len(part)} were left"
        )
    return part


def _read_dims(file, path: Path, expected_ndims: int) -> tuple[list[int], int]:
    """Valida la cabecera del fichero abierto y devuelve (dims, tamano de registro)."""
    (magic,) = struct.unpack(">I", _read_header_part(file, 4, path))
    if magic >> 16:
        raise ValueError(f"{path}: IDX magic must start with two zero bytes, got {magic:#010x}")
    data_type = (magic >> 8) & 0xFF
    if data_type not in _ELEMENT_SIZE_BY_TYPE:
        supported = ", ".join(f"{t:#04x}" for t in _ELEMENT_SIZE_BY_TYPE)
        raise ValueError(
            f"{path}: unsupported IDX data type {data_type:#04x}, supported: {supported}"
        )
    ndims = magic & 0xFF
    if ndims != expected_ndims:
        raise ValueError(
            f"{path}: expected {expected_ndims} dimension(s), the magic declares {ndims}"
        )
    dims = list(struct.unpack(f">{ndims}I", _read_header_part(file, 4 * ndims, path)))

    # Las dimensiones salen de un fichero que aun no sabemos si es fiable: se contrastan
    # con el tamano real antes de que nada las use para posicionarse.
    record_size = math.prod(dims[1:]) * _ELEMENT_SIZE_BY_TYPE[data_type]
    expected_size = 4 + 4 * ndims + dims[0] * record_size
    actual_size = os.fstat(file.fileno()).st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: the file has {actual_size} bytes but its header implies {expected_size}: "
            "truncated or with extra bytes"
        )
    return dims, record_size


class IdxRawSampleSequence:
    def __init__(self, images_path: Path, labels_path: Path):
        # Si algo falla mientras se abre y valida, el `with` cierra lo que ya este abierto
        # (uno o los dos ficheros). Si todo sale bien, pop_all() traspasa esa
        # responsabilidad a self._stack y el `with` ya no cierra nada.
        with ExitStack() as stack:
            images = stack.enter_context(open(images_path, "rb"))
            labels = stack.enter_context(open(labels_path, "rb"))
            images_dims, self._images_record_size = _read_dims(images, images_path, _IMAGES_NDIMS)
            labels_dims, self._labels_record_size = _read_dims(labels, labels_path, _LABELS_NDIMS)
            images_n, *x_shape = images_dims
            (labels_n,) = labels_dims
            if images_n != labels_n:
                raise ValueError(
                    f"the number of samples differs: {images_path} has {images_n}, "
                    f"{labels_path} has {labels_n}"
                )
            self._images = images
            self._labels = labels
            self._n = images_n
            self._x_shape = tuple(x_shape)
            self._stack = stack.pop_all()

    def __len__(self) -> int:
        return self._n

    @property
    def x_shape(self) -> tuple[int, ...]:
        return self._x_shape

    def __getitem__(self, i: int) -> RawSample:
        if not 0 <= i < self._n:
            raise IndexError(f"sample index {i} is out of range for {self._n} samples")
        # Cada registro esta en cabecera + i * tamano_de_registro; la cabecera mide
        # 4 bytes de magic mas 4 por dimension.
        self._images.seek(4 + 4 * _IMAGES_NDIMS + i * self._images_record_size)
        x = self._images.read(self._images_record_size)
        self._labels.seek(4 + 4 * _LABELS_NDIMS + i * self._labels_record_size)
        y_true = self._labels.read(self._labels_record_size)
        return RawSample(x=x, y_true=y_true)
