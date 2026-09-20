import struct
from pathlib import Path

_IMAGES_NDIMS = 3  # el ndims esperado viene de fuera del fichero: 3 para imagenes


class IdxRawSampleSequence:
    def __init__(self, images_path: Path, labels_path: Path):
        with open(images_path, "rb") as images:
            _magic, _n, *x_shape = struct.unpack(f">II{_IMAGES_NDIMS - 1}I", images.read(4 + 4 * _IMAGES_NDIMS))
        self._x_shape = tuple(x_shape)

        with open(labels_path, "rb") as labels:
            _magic, self._n = struct.unpack(">II", labels.read(8))

    def __len__(self) -> int:
        return self._n

    @property
    def x_shape(self) -> tuple[int, ...]:
        return self._x_shape
