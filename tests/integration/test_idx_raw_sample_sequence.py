from mnist.adapters.idx_raw_sample_sequence import IdxRawSampleSequence
from tests.idx_files import write_idx


def test_len_sale_del_n_de_la_cabecera_de_etiquetas(tmp_path):
    # 256 = 00 00 01 00 en big-endian: si N se leyera en little-endian saldria 65536
    images = write_idx(tmp_path, "images", dims=(256, 2, 3))
    labels = write_idx(tmp_path, "labels", dims=(256,))

    assert len(IdxRawSampleSequence(images, labels)) == 256
