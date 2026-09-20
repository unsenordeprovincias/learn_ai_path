"""Bateria de contrato del port RawSampleSequence.

Cada adaptador aporta una fabrica: dados los ejemplos en abstracto (imagenes, etiquetas
y forma de la entrada), construye SU fuente de datos y la entrega dentro de un `with`.
Asi el cierre de recursos queda en la fabrica, o sea en el adaptador, y no en el contrato:
el port solo declara len, x_shape y x[i]. Un adaptador nuevo se registra en ADAPTERS y
pasa exactamente los mismos tests.
"""
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, ContextManager, Iterator

import pytest

from mnist.adapters.idx_raw_sample_sequence import IdxRawSampleSequence
from mnist.domain.samples import RawSample
from mnist.ports.raw_sample_sequence import RawSampleSequence
from tests.idx_files import write_idx
from tests.in_memory_raw_sample_sequence import InMemoryRawSampleSequence

Build = Callable[
    [Path, list[bytes], list[bytes], tuple[int, ...]], ContextManager[RawSampleSequence]
]


@contextmanager
def _idx_seek_read(
    tmp_path: Path, images: list[bytes], labels: list[bytes], x_shape: tuple[int, ...]
) -> Iterator[RawSampleSequence]:
    n = len(images)
    images_path = write_idx(tmp_path, "images", dims=(n, *x_shape), content=b"".join(images))
    labels_path = write_idx(tmp_path, "labels", dims=(n,), content=b"".join(labels))
    with IdxRawSampleSequence(images_path, labels_path) as sequence:
        yield sequence


@contextmanager
def _in_memory(
    tmp_path: Path, images: list[bytes], labels: list[bytes], x_shape: tuple[int, ...]
) -> Iterator[RawSampleSequence]:
    # sin ficheros ni cabecera: no usa tmp_path y no tiene nada que cerrar
    samples = [RawSample(x=x, y_true=y_true) for x, y_true in zip(images, labels)]
    yield InMemoryRawSampleSequence(samples, x_shape)


ADAPTERS = {
    "idx-seek+read": _idx_seek_read,
    "in-memory": _in_memory,
}


@pytest.fixture(params=list(ADAPTERS.values()), ids=list(ADAPTERS))
def build(request) -> Build:
    return request.param


def _distinguible():
    """N=3 e imagenes (2, 4): ningun byte de imagen se repite entre registros.

    Asi un desplazamiento mal calculado devuelve bytes de otro registro y se nota.
    """
    x_shape = (2, 4)
    tamano = math.prod(x_shape)
    imagenes = [bytes(k * tamano + j for j in range(tamano)) for k in range(3)]
    etiquetas = [bytes([5]), bytes([0]), bytes([9])]
    return imagenes, etiquetas, x_shape


def test_len_es_el_numero_de_ejemplos(tmp_path, build):
    # 256 = 00 00 01 00 en big-endian: un adaptador que decodifique N en little-endian
    # obtendria 65536. Con otro formato el numero es simplemente uno mas.
    imagenes = [bytes(6)] * 256
    etiquetas = [bytes([i % 10]) for i in range(256)]

    with build(tmp_path, imagenes, etiquetas, (2, 3)) as sequence:
        assert len(sequence) == 256


def test_x_shape_es_la_forma_de_un_ejemplo_de_entrada_sin_el_n(tmp_path, build):
    # las tres dimensiones son distintas para que un orden equivocado o un recorte
    # mal hecho no pase por casualidad
    imagenes = [bytes(3 * 40)] * 2

    with build(tmp_path, imagenes, [bytes([0]), bytes([1])], (3, 40)) as sequence:
        assert sequence.x_shape == (3, 40)


def test_x_de_i_devuelve_el_registro_i_como_raw_sample(tmp_path, build):
    imagenes, etiquetas, x_shape = _distinguible()

    with build(tmp_path, imagenes, etiquetas, x_shape) as sequence:
        assert sequence[1] == RawSample(x=imagenes[1], y_true=etiquetas[1])


def test_cada_x_de_i_empareja_la_imagen_i_con_la_etiqueta_i(tmp_path, build):
    imagenes, etiquetas, x_shape = _distinguible()

    with build(tmp_path, imagenes, etiquetas, x_shape) as sequence:
        esperado = [RawSample(x=im, y_true=et) for im, et in zip(imagenes, etiquetas)]
        assert [sequence[i] for i in range(len(sequence))] == esperado


def test_los_bordes_0_y_n_menos_1_son_validos(tmp_path, build):
    imagenes, etiquetas, x_shape = _distinguible()

    with build(tmp_path, imagenes, etiquetas, x_shape) as sequence:
        assert sequence[0] == RawSample(x=imagenes[0], y_true=etiquetas[0])
        assert sequence[len(sequence) - 1] == RawSample(x=imagenes[-1], y_true=etiquetas[-1])


@pytest.mark.parametrize("i", [-1, -3, 3, 4], ids=["-1", "-n", "n", "n+1"])
def test_indice_fuera_de_0_n_lanza_index_error_negativos_incluidos(tmp_path, build, i):
    # n = 3. -1 y -n serian validos en una secuencia de Python; aqui no, a proposito
    imagenes, etiquetas, x_shape = _distinguible()

    with build(tmp_path, imagenes, etiquetas, x_shape) as sequence:
        with pytest.raises(IndexError):
            sequence[i]


def test_leer_la_misma_posicion_dos_veces_da_lo_mismo_aunque_haya_otras_lecturas_en_medio(
    tmp_path, build
):
    # sin estado de recorrido: leer x[i] no puede depender de lo que se leyo antes
    imagenes, etiquetas, x_shape = _distinguible()

    with build(tmp_path, imagenes, etiquetas, x_shape) as sequence:
        primera = sequence[2]
        sequence[0]
        segunda = sequence[2]

    assert primera == segunda == RawSample(x=imagenes[2], y_true=etiquetas[2])
