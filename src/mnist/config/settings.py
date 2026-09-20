import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

# Donde estan hoy los ficheros IDX descomprimidos (git los ignora, salen del .gz). Se deriva
# de la ubicacion de este fichero (repo/src/mnist/config/) y no del directorio desde el que
# se lance nada.
_MNIST_DIR = Path(__file__).resolve().parents[3] / "playgrounds" / "original_mnist" / "datasets"


@dataclass(frozen=True)
class Settings:
    train_images_path: Path = _MNIST_DIR / "train-images-idx3-ubyte"
    train_labels_path: Path = _MNIST_DIR / "train-labels-idx1-ubyte"
    t10k_images_path: Path = _MNIST_DIR / "t10k-images-idx3-ubyte"
    t10k_labels_path: Path = _MNIST_DIR / "t10k-labels-idx1-ubyte"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Cada campo se puede sobrescribir con la variable de entorno del mismo nombre en
        mayusculas (p. ej. TRAIN_IMAGES_PATH). Las demas variables se ignoran."""
        environ = os.environ if environ is None else environ
        overrides = {
            field.name: Path(environ[field.name.upper()])
            for field in fields(cls)
            if field.name.upper() in environ
        }
        return cls(**overrides)


settings = Settings.from_env()
