"""Escribe CSV sinteticos de MNIST para los tests: una fila por ejemplo, la etiqueta en la
primera columna y despues los pixeles, enteros separados por comas y sin cabecera."""
from pathlib import Path


def csv_bytes(images: list[bytes], labels: list[bytes], newline: bytes = b"\n") -> bytes:
    """Contenido del CSV. Cada etiqueta es un bytes de longitud 1, cada imagen sus pixeles."""
    rows = [",".join(str(v) for v in label + image).encode() for image, label in zip(images, labels)]
    return b"".join(row + newline for row in rows)


def write_csv(
    directory: Path,
    name: str,
    images: list[bytes],
    labels: list[bytes],
    newline: bytes = b"\n",
) -> Path:
    path = directory / name
    path.write_bytes(csv_bytes(images, labels, newline))
    return path
