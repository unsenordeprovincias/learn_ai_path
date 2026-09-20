from pydantic_settings import BaseSettings
from pathlib import Path

# Donde estan hoy los ficheros IDX descomprimidos (git los ignora, salen del .gz). Se deriva
# de la ubicacion de este fichero (repo/src/mnist/config/) y no del directorio desde el que
# se lance nada. Cada ruta se puede sobrescribir con una variable de entorno, p. ej.
# TRAIN_IMAGES_PATH.
_MNIST_DIR = Path(__file__).resolve().parents[3] / "playgrounds" / "original_mnist" / "datasets"

class Settings(BaseSettings):
    app_name: str = "nombre_proyecto"
    debug: bool = False
    log_level: str = "INFO"

    train_images_path: Path = _MNIST_DIR / "train-images-idx3-ubyte"
    train_labels_path: Path = _MNIST_DIR / "train-labels-idx1-ubyte"
    t10k_images_path: Path = _MNIST_DIR / "t10k-images-idx3-ubyte"
    t10k_labels_path: Path = _MNIST_DIR / "t10k-labels-idx1-ubyte"

    class Config:
        env_file = Path(__file__).parent.parent / ".env.dev"
        case_sensitive = False

settings = Settings()