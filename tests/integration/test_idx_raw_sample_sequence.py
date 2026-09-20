import gzip
import io
import math

import pytest

from mnist.adapters.idx_raw_sample_sequence import IdxRawSampleSequence
from tests.idx_files import write_idx


def test_len_sale_del_n_de_la_cabecera_de_etiquetas(tmp_path):
    # 256 = 00 00 01 00 en big-endian: si N se leyera en little-endian saldria 65536
    images = write_idx(tmp_path, "images", dims=(256, 2, 3))
    labels = write_idx(tmp_path, "labels", dims=(256,))

    assert len(IdxRawSampleSequence(images, labels)) == 256


def test_x_shape_sale_de_la_cabecera_de_imagenes_sin_el_n(tmp_path):
    # las tres dimensiones son distintas para que un orden equivocado o un recorte
    # mal hecho de dims no pase por casualidad
    images = write_idx(tmp_path, "images", dims=(2, 3, 40))
    labels = write_idx(tmp_path, "labels", dims=(2,))

    assert IdxRawSampleSequence(images, labels).x_shape == (3, 40)


# ---------- validaciones al abrir ----------
# Cada test rompe UNA cosa de un par valido (N=2, imagenes (3, 40)) y comprueba que el
# error nombra esa cosa: un ValueError cualquiera podria venir de otra validacion.

def _par_valido(tmp_path):
    images = write_idx(tmp_path, "images", dims=(2, 3, 40))
    labels = write_idx(tmp_path, "labels", dims=(2,))
    return images, labels


@pytest.mark.parametrize("roto", ["images", "labels"])
def test_magic_que_no_empieza_por_dos_ceros_se_rechaza(tmp_path, roto):
    images, labels = _par_valido(tmp_path)
    path = {"images": images, "labels": labels}[roto]
    contenido = bytearray(path.read_bytes())
    contenido[0] = 0xFF
    path.write_bytes(contenido)

    with pytest.raises(ValueError, match="magic"):
        IdxRawSampleSequence(images, labels)


@pytest.mark.parametrize("roto, dims", [("images", (2, 3, 40)), ("labels", (2,))])
def test_tipo_distinto_de_unsigned_byte_se_rechaza(tmp_path, roto, dims):
    images, labels = _par_valido(tmp_path)
    # 0x09 = signed byte: tambien ocupa 1 byte, asi que solo cambia el tipo, no el tamano
    write_idx(tmp_path, roto, dims=dims, data_type=0x09)

    with pytest.raises(ValueError, match="data type"):
        IdxRawSampleSequence(images, labels)


@pytest.mark.parametrize(
    "roto, dims",
    [("images", (2, 7)), ("images", (2, 3, 4, 5)), ("labels", (2, 5))],
    ids=["imagenes ndims=2", "imagenes ndims=4", "etiquetas ndims=2"],
)
def test_ndims_distinto_del_esperado_se_rechaza(tmp_path, roto, dims):
    images, labels = _par_valido(tmp_path)
    write_idx(tmp_path, roto, dims=dims)

    with pytest.raises(ValueError, match="dimension"):
        IdxRawSampleSequence(images, labels)


@pytest.mark.parametrize(
    "roto, dims, sobran",
    [
        ("images", (2, 3, 40), -1),
        ("images", (2, 3, 40), +1),
        ("labels", (2,), -1),
        ("labels", (2,), +1),
    ],
    ids=["imagenes truncadas", "imagenes con bytes de sobra",
         "etiquetas truncadas", "etiquetas con bytes de sobra"],
)
def test_tamano_de_fichero_incoherente_con_la_cabecera_se_rechaza(tmp_path, roto, dims, sobran):
    images, labels = _par_valido(tmp_path)
    # la cabecera sigue declarando las mismas dimensiones; solo cambia el contenido
    write_idx(tmp_path, roto, dims=dims, content=bytes(max(0, math.prod(dims) + sobran)))

    with pytest.raises(ValueError, match="bytes"):
        IdxRawSampleSequence(images, labels)


@pytest.mark.parametrize("roto", ["images", "labels"])
@pytest.mark.parametrize("bytes_que_quedan", [0, 3, 6], ids=["vacio", "magic cortado", "dimensiones cortadas"])
def test_fichero_mas_corto_que_su_cabecera_se_rechaza(tmp_path, roto, bytes_que_quedan):
    images, labels = _par_valido(tmp_path)
    path = {"images": images, "labels": labels}[roto]
    path.write_bytes(path.read_bytes()[:bytes_que_quedan])

    with pytest.raises(ValueError, match="header"):
        IdxRawSampleSequence(images, labels)


def test_n_distinto_entre_imagenes_y_etiquetas_se_rechaza(tmp_path):
    # cada fichero es coherente consigo mismo: solo discrepan entre si
    images = write_idx(tmp_path, "images", dims=(3, 3, 40))
    labels = write_idx(tmp_path, "labels", dims=(2,))

    with pytest.raises(ValueError, match="differs"):
        IdxRawSampleSequence(images, labels)


def test_etiquetas_con_forma_n_por_uno_se_rechazan(tmp_path):
    # (N, 1) tiene el mismo N y los mismos bytes que (N,), pero no es la forma exacta
    images, labels = _par_valido(tmp_path)
    write_idx(tmp_path, "labels", dims=(2, 1))

    with pytest.raises(ValueError, match="dimension"):
        IdxRawSampleSequence(images, labels)


@pytest.mark.parametrize("roto", ["images", "labels"])
def test_un_fichero_gz_se_rechaza_por_su_magic(tmp_path, roto):
    images, labels = _par_valido(tmp_path)
    path = {"images": images, "labels": labels}[roto]
    buffer = io.BytesIO()
    # con filename, gzip pone el flag FNAME y la cabecera empieza por 1f 8b 08 08,
    # como los .gz de MNIST creados con la linea de comandos
    with gzip.GzipFile(filename=path.name, mode="wb", fileobj=buffer) as gz:
        gz.write(path.read_bytes())
    path.write_bytes(buffer.getvalue())
    assert path.read_bytes()[:4] == bytes.fromhex("1f8b0808")

    with pytest.raises(ValueError, match="0x1f8b0808"):
        IdxRawSampleSequence(images, labels)
