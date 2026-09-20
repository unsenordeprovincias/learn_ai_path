import struct
from pathlib import Path


class IdxRawSampleSequence:
    def __init__(self, images_path: Path, labels_path: Path):
        with open(labels_path, "rb") as labels:
            _magic, self._n = struct.unpack(">II", labels.read(8))

    def __len__(self) -> int:
        return self._n
