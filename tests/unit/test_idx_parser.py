# tests/unit/test_idx_parser.py
import struct
import pytest
from mnist.adapters.idx_mnist_source import parse_idx_header, parse_idx_images, parse_idx_labels


# ---------- Nivel 1: header ----------

def test_header_imagenes_se_interpreta_correctamente():
    header = struct.pack('>IIII', 0x00000803, 2, 28, 28)
    data_type, num_dims, dims = parse_idx_header(header)
    assert data_type == 0x08
    assert num_dims == 3
    assert dims == [2, 28, 28]


def test_header_labels_se_interpreta_correctamente():
    header = struct.pack('>II', 0x00000801, 2)
    data_type, num_dims, dims = parse_idx_header(header)
    assert data_type == 0x08
    assert num_dims == 1
    assert dims == [2]


def test_magic_number_invalido_lanza_error():
    header = struct.pack('>II', 0x00000000, 2)
    with pytest.raises(ValueError):
        parse_idx_header(header)


@pytest.mark.parametrize("byte0, byte1", [(1, 0), (0, 1)])
def test_magic_con_los_dos_primeros_bytes_distintos_de_cero_lanza_error(byte0, byte1):
    """El formato IDX reserva los dos primeros bytes del magic number a cero."""
    header = struct.pack('>BBBBI', byte0, byte1, 0x08, 1, 2)
    with pytest.raises(ValueError, match="two zero bytes"):
        parse_idx_header(header)


@pytest.mark.parametrize("data_type", [0x09, 0x0B, 0x0C, 0x0D, 0x0E])
def test_tipo_de_dato_distinto_de_ubyte_lanza_error(data_type):
    """Los parsers leen con 'B' (unsigned byte): cualquier otro tipo
    (signed byte, short, int, float, double) daría valores basura."""
    header = struct.pack('>BBBBI', 0, 0, data_type, 1, 2)
    with pytest.raises(ValueError, match="data type"):
        parse_idx_header(header)


@pytest.mark.parametrize("header", [
    struct.pack('>BBBBI', 0, 0, 0x08, 3, 2),            # declara 3 dimensiones, trae 1
    struct.pack('>BBBBIII', 0, 0, 0x08, 1, 2, 28, 28),  # declara 1 dimension, trae 3
])
def test_num_dims_declarado_no_coincide_con_las_dimensiones_leidas(header):
    with pytest.raises(ValueError, match="dimension"):
        parse_idx_header(header)


# ---------- Nivel 2: una sola imagen / un solo label ----------

def _fabricar_idx_imagenes(pixel_matrices):
    n = len(pixel_matrices)
    body = struct.pack('>IIII', 0x00000803, n, 28, 28)
    for matriz in pixel_matrices:
        for fila in matriz:
            body += bytes(fila)
    return body

def _fabricar_idx_labels(labels):
    body = struct.pack('>II', 0x00000801, len(labels))
    body += bytes(labels)
    return body

def test_una_imagen_se_extrae_con_valores_correctos():
    diagonal = [[255 if i == j else 0 for j in range(28)] for i in range(28)]
    source = _fabricar_idx_imagenes([diagonal])

    imagenes = list(parse_idx_images(source))

    assert len(imagenes) == 1
    assert imagenes[0][0] == 255
    for i in range(28):
        assert imagenes[0][28 * i + i] == 255 
        if 28 * i + i != 28 * i:
            assert imagenes[0][28 * i ] == 0 
                

def test_una_sola_label_se_extrae_con_valor_correcto(tmp_path):
    source = _fabricar_idx_labels([7])

    labels = list(parse_idx_labels(source))

    assert labels == [7]


# ---------- Nivel 3: dos imágenes / dos labels ----------

def test_dos_imagenes_no_se_mezclan():
    ceros = [[0] * 28 for _ in range(28)]
    llena = [[255] * 28 for _ in range(28)]
    source = _fabricar_idx_imagenes([ceros, llena])

    imagenes = list(parse_idx_images(source))

    assert imagenes[0] == tuple([0] * 28 * 28)
    assert imagenes[1] == tuple([255] * 28 * 28)
    assert imagenes[0] is not imagenes[1]


def test_dos_labels_mantienen_orden():
    path = _fabricar_idx_labels([3, 8])

    labels = list(parse_idx_labels(path))

    assert labels == [3, 8]


# ---------- Nivel 4: fichero equivocado ----------

def test_fichero_de_labels_pasado_como_imagenes_lanza_error():
    # 8 labels -> 16 bytes, los justos para que se pueda leer una cabecera de imágenes
    source = _fabricar_idx_labels([1, 2, 3, 4, 5, 6, 7, 8])

    with pytest.raises(ValueError):
        list(parse_idx_images(source))


def test_fichero_de_imagenes_pasado_como_labels_lanza_error():
    source = _fabricar_idx_imagenes([[[0] * 28 for _ in range(28)]])

    with pytest.raises(ValueError):
        list(parse_idx_labels(source))