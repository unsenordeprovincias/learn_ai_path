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
