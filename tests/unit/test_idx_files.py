import math

import pytest

from tests.idx_files import write_idx

# (dims, tamano esperado del fichero). Los tamanos son literales calculados a mano,
# no salen de la formula del helper: el helper no puede ser su propio oraculo.
CASOS = {
    "imagenes (2, 3, 40)": ((2, 3, 40), 16 + 2 * 3 * 40),  # 256
    "etiquetas N=256": ((256,), 8 + 256),  # 264
    "ndims=2": ((4, 7), 12 + 4 * 7),  # 40
    "ndims=4": ((2, 3, 4, 5), 20 + 2 * 3 * 4 * 5),  # 140
}


@pytest.mark.parametrize("dims, tamano", CASOS.values(), ids=CASOS.keys())
def test_el_tamano_del_fichero_coincide_con_la_formula(tmp_path, dims, tamano):
    path = write_idx(tmp_path, "datos", dims=dims)

    cabecera = 4 + 4 * len(dims)
    assert path.stat().st_size == tamano
    assert path.stat().st_size == cabecera + math.prod(dims)


def test_cabecera_de_imagenes_es_big_endian_y_codifica_tipo_y_ndims(tmp_path):
    path = write_idx(tmp_path, "imagenes", dims=(2, 3, 40))

    # magic 00 00 08 03 | 2 | 3 | 40 (= 0x28), cada dimension en 4 bytes
    assert path.read_bytes()[:16] == bytes.fromhex("00000803" "00000002" "00000003" "00000028")


def test_cabecera_de_etiquetas_con_n_256_delata_un_error_de_endianness(tmp_path):
    path = write_idx(tmp_path, "etiquetas", dims=(256,))

    # 256 en big-endian es 00 00 01 00; en little-endian seria 00 01 00 00
    assert path.read_bytes()[:8] == bytes.fromhex("00000801" "00000100")


def test_ndims_del_magic_sale_de_las_dimensiones(tmp_path):
    for dims in [(4, 7), (2, 3, 4, 5)]:
        path = write_idx(tmp_path, f"ndims{len(dims)}", dims=dims)
        assert path.read_bytes()[3] == len(dims)


def test_el_tipo_se_escribe_en_el_tercer_byte_del_magic(tmp_path):
    path = write_idx(tmp_path, "con_signo", dims=(3,), data_type=0x09)

    assert path.read_bytes()[:4] == bytes.fromhex("00000901")


def test_el_contenido_se_escribe_tal_cual_tras_la_cabecera(tmp_path):
    path = write_idx(tmp_path, "etiquetas", dims=(3,), content=bytes([7, 8, 9]))

    assert path.read_bytes()[8:] == bytes([7, 8, 9])


def test_el_contenido_no_se_valida_contra_las_dimensiones(tmp_path):
    # hace falta poder fabricar ficheros incoherentes para probar sus rechazos
    truncado = write_idx(tmp_path, "truncado", dims=(2, 3, 40), content=b"abc")
    de_sobra = write_idx(tmp_path, "de_sobra", dims=(3,), content=bytes(10))

    assert truncado.stat().st_size == 16 + 3
    assert de_sobra.stat().st_size == 8 + 10


def test_sin_contenido_los_bytes_por_defecto_no_son_todos_iguales(tmp_path):
    path = write_idx(tmp_path, "datos", dims=(3,))

    assert path.read_bytes()[8:] == bytes([0, 1, 2])
