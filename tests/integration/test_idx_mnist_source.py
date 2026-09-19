# tests/integration/test_idx_mnist_source.py
import struct
from mnist.adapters.idx_mnist_source import IdxMnistSource
from mnist.domain.models import Sample, Vector
import pytest



def _fabricar_par(patrones_y_labels):
    n = len(patrones_y_labels)

    body_img = struct.pack('>IIII', 0x00000803, n, 28, 28)
    body_lbl = struct.pack('>II', 0x00000801, n)

    for patron, label in patrones_y_labels:
        body_img += bytes(patron)
        body_lbl += bytes([label])

    return body_img, body_lbl


def test_numero_de_samples_coincide_con_numero_de_labels():
    patron_a = [0] * 28 * 28 
    patron_b = [255] * 28 * 28 
    img_source, lbl_source = _fabricar_par([(patron_a, 5), (patron_b, 9)])

    samples = list(IdxMnistSource(img_source, lbl_source).load())

    assert len(samples) == 2
    assert all(isinstance(s, Sample) for s in samples)


def test_cada_sample_empareja_imagen_con_su_label_correcto():
    patron_a = [0] * 28 * 28
    patron_b = [255] * 28 * 28
    img_source, lbl_source = _fabricar_par([(patron_a, 5), (patron_b, 9)])

    samples = list(IdxMnistSource(img_source, lbl_source).load())

    assert samples[0].x == Vector(patron_a)
    assert samples[0].y_true == Vector[0,0,0,0,0,1,0,0,0,0]
    assert samples[1].x == Vector([1]*28*28)
    assert samples[1].y_true == Vector[0,0,0,0,0,0,0,0,0,1]


def test_desajuste_entre_n_imagenes_y_n_labels_lanza_error():
    patron_a = [0] * 28 * 28
    patron_b = [255] * 28 * 28

    # dos imágenes fabricadas a mano, pero solo un label -> ficheros truncados/corruptos
    img_source, _ = _fabricar_par([(patron_a, 0), (patron_b, 0)])
    _, lbl_source = _fabricar_par([(patron_a, 5)])

    with pytest.raises(ValueError):
        IdxMnistSource(img_source, lbl_source).load()


def test_ficheros_intercambiados_lanzan_error_al_construir_la_fuente():
    # 8 muestras: el fichero de labels (16 bytes) tiene el tamaño justo de una cabecera de imágenes
    img_source, lbl_source = _fabricar_par([([0] * 28 * 28, label) for label in range(8)])

    with pytest.raises(ValueError):
        IdxMnistSource(lbl_source, img_source)


# ---------- tamaño del cuerpo frente a lo que declara la cabecera ----------
# Un .gz descargado a medias tiene la cabecera bien y le faltan datos: debe
# fallar al construir la fuente, no a mitad de una época.

def _par_de_dos():
    return _fabricar_par([([0] * 28 * 28, 1), ([255] * 28 * 28, 2)])


def test_fichero_de_imagenes_truncado_lanza_error_al_construir():
    img_source, lbl_source = _par_de_dos()

    with pytest.raises(ValueError, match="images file"):
        IdxMnistSource(img_source[:-10], lbl_source)


def test_fichero_de_labels_truncado_lanza_error_al_construir():
    img_source, lbl_source = _par_de_dos()

    with pytest.raises(ValueError, match="labels file"):
        IdxMnistSource(img_source, lbl_source[:-1])


def test_fichero_de_imagenes_con_bytes_sobrantes_lanza_error_al_construir():
    img_source, lbl_source = _par_de_dos()

    with pytest.raises(ValueError, match="images file"):
        IdxMnistSource(img_source + b"\x00", lbl_source)


def test_fichero_de_labels_con_bytes_sobrantes_lanza_error_al_construir():
    img_source, lbl_source = _par_de_dos()

    with pytest.raises(ValueError, match="labels file"):
        IdxMnistSource(img_source, lbl_source + b"\x00")