"""Escribe ficheros IDX sinteticos para los tests.

Es una segunda implementacion del formato, escrita por quien tambien escribe el
adaptador: si ambos comparten un malentendido, los tests sinteticos pasan igual.
Lo que lo cubre son los ficheros reales (test de integracion), no este helper.
"""
import math
import struct
from pathlib import Path

UBYTE = 0x08


def write_idx(
    directory: Path,
    name: str,
    dims: tuple[int, ...],
    data_type: int = UBYTE,
    content: bytes | None = None,
) -> Path:
    """Escribe directory/name con cabecera IDX y devuelve su ruta.

    Cabecera: magic de 4 bytes (0, 0, data_type, ndims) y luego una dimension por
    cada entero de 4 bytes, todo big-endian. ndims sale de len(dims).

    content va tal cual tras la cabecera y NO se valida contra dims: los tests
    necesitan fabricar ficheros truncados o con bytes de sobra. Si es None se
    rellena con prod(dims) bytes (i % 256), o sea 1 byte por elemento, lo cual
    solo es cierto para tipos de 1 byte; con otro tipo hay que pasar content.
    """
    if content is None:
        content = bytes(i % 256 for i in range(math.prod(dims)))

    magic = struct.pack(">BBBB", 0, 0, data_type, len(dims))
    dimensions = struct.pack(f">{len(dims)}I", *dims)

    path = directory / name
    path.write_bytes(magic + dimensions + content)
    return path
