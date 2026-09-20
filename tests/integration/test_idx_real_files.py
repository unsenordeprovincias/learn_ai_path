"""El adaptador IDX contra los ficheros reales de MNIST (descomprimidos).

Los tests sinteticos usan un helper y un adaptador escritos por la misma persona: si ambos
comparten un malentendido del formato, pasan igual. Aqui los literales salen de fuera:
los tamanos con `stat`, y las cabeceras y las primeras etiquetas con un `xxd` hecho a mano.
Las rutas vienen de la configuracion; si faltan, el test se salta y dice que ruta busco.
"""
import pytest

from mnist.adapters.idx_raw_sample_sequence import IdxRawSampleSequence
from mnist.config.settings import settings

# split -> (n, tamano del fichero de imagenes, tamano del de etiquetas, primeras 8 etiquetas)
CASES = {
    "train": (60_000, 47_040_016, 60_008, [5, 0, 4, 1, 9, 2, 1, 3]),
    "t10k": (10_000, 7_840_016, 10_008, [7, 2, 1, 0, 4, 1, 4, 9]),
}


def _paths(split):
    paths = []
    for kind in ("images", "labels"):
        field = f"{split}_{kind}_path"
        path = getattr(settings, field)
        if not path.is_file():
            pytest.skip(
                f"real MNIST file not found: {path} "
                f"(set the {field.upper()} environment variable to its location)"
            )
        paths.append(path)
    return paths


@pytest.mark.parametrize("split", CASES)
def test_los_ficheros_reales_tienen_el_tamano_esperado(split):
    _, images_size, labels_size, _ = CASES[split]
    images, labels = _paths(split)

    assert images.stat().st_size == images_size
    assert labels.stat().st_size == labels_size


@pytest.mark.parametrize("split", CASES)
def test_len_y_forma_salen_de_las_cabeceras_reales(split):
    n, _, _, _ = CASES[split]

    with IdxRawSampleSequence(*_paths(split)) as sequence:
        assert len(sequence) == n
        assert sequence.x_shape == (28, 28)


@pytest.mark.parametrize("split", CASES)
def test_las_primeras_etiquetas_coinciden_con_las_del_hexdump(split):
    _, _, _, primeras = CASES[split]

    with IdxRawSampleSequence(*_paths(split)) as sequence:
        assert [sequence[i].y_true for i in range(8)] == [bytes([v]) for v in primeras]


@pytest.mark.parametrize("split", CASES)
def test_la_primera_y_la_ultima_muestra_se_leen_con_784_bytes_de_imagen(split):
    n, _, _, _ = CASES[split]

    with IdxRawSampleSequence(*_paths(split)) as sequence:
        assert len(sequence[0].x) == 28 * 28
        assert len(sequence[n - 1].x) == 28 * 28
