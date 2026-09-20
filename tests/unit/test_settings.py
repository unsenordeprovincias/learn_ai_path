from pathlib import Path

import pytest

from mnist.config.settings import Settings

# La raiz del repo se deriva aqui desde la ubicacion de este test (tests/unit/), no con la
# misma cuenta que usa settings.py: asi el test no es su propio oraculo.
DATASETS = Path(__file__).resolve().parents[2] / "playgrounds" / "original_mnist" / "datasets"

FIELDS = {
    "train_images_path": "train-images-idx3-ubyte",
    "train_labels_path": "train-labels-idx1-ubyte",
    "t10k_images_path": "t10k-images-idx3-ubyte",
    "t10k_labels_path": "t10k-labels-idx1-ubyte",
}


@pytest.mark.parametrize("field, filename", FIELDS.items())
def test_sin_variables_de_entorno_cada_ruta_apunta_a_donde_estan_los_ficheros(field, filename):
    assert getattr(Settings.from_env({}), field) == DATASETS / filename


@pytest.mark.parametrize("field", FIELDS)
def test_una_variable_de_entorno_sobrescribe_solo_su_ruta(field):
    settings = Settings.from_env({field.upper(): "/otro/sitio/fichero"})

    assert getattr(settings, field) == Path("/otro/sitio/fichero")
    for other, filename in FIELDS.items():
        if other != field:
            assert getattr(settings, other) == DATASETS / filename


def test_las_variables_de_entorno_que_no_son_de_la_configuracion_se_ignoran():
    settings = Settings.from_env({"HOME": "/home/alguien", "PATH": "/usr/bin"})

    assert settings == Settings.from_env({})
