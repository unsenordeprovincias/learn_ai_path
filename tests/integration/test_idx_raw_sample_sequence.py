import gzip
import io
import math

import pytest

import mnist.adapters.idx_raw_sample_sequence as adaptador
from mnist.adapters.idx_raw_sample_sequence import IdxRawSampleSequence
from mnist.domain.samples import RawSample
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


# ---------- acceso por posicion: x[i] ----------

def _par_distinguible(tmp_path):
    """N=3 e imagenes (2, 4): ningun byte de imagen se repite entre registros.

    Asi un desplazamiento mal calculado devuelve bytes de otro registro y se nota.
    """
    n, x_shape = 3, (2, 4)
    tamano = math.prod(x_shape)
    imagenes = [bytes(k * tamano + j for j in range(tamano)) for k in range(n)]
    etiquetas = [bytes([5]), bytes([0]), bytes([9])]
    images = write_idx(tmp_path, "images", dims=(n, *x_shape), content=b"".join(imagenes))
    labels = write_idx(tmp_path, "labels", dims=(n,), content=b"".join(etiquetas))
    return images, labels, imagenes, etiquetas


def test_x_de_i_devuelve_el_registro_i_como_raw_sample(tmp_path):
    images, labels, imagenes, etiquetas = _par_distinguible(tmp_path)

    sample = IdxRawSampleSequence(images, labels)[1]

    assert sample == RawSample(x=imagenes[1], y_true=etiquetas[1])


def test_cada_x_de_i_empareja_la_imagen_i_con_la_etiqueta_i(tmp_path):
    images, labels, imagenes, etiquetas = _par_distinguible(tmp_path)
    sequence = IdxRawSampleSequence(images, labels)

    esperado = [RawSample(x=im, y_true=et) for im, et in zip(imagenes, etiquetas)]
    assert [sequence[i] for i in range(len(sequence))] == esperado


def test_los_bordes_0_y_n_menos_1_son_validos(tmp_path):
    images, labels, imagenes, etiquetas = _par_distinguible(tmp_path)
    sequence = IdxRawSampleSequence(images, labels)

    assert sequence[0] == RawSample(x=imagenes[0], y_true=etiquetas[0])
    assert sequence[len(sequence) - 1] == RawSample(x=imagenes[-1], y_true=etiquetas[-1])


@pytest.mark.parametrize("i", [-1, -3, 3, 4], ids=["-1", "-n", "n", "n+1"])
def test_indice_fuera_de_0_n_lanza_index_error_negativos_incluidos(tmp_path, i):
    # n = 3. -1 y -n serian validos en una secuencia de Python; aqui no, a proposito
    images, labels, _, _ = _par_distinguible(tmp_path)

    with pytest.raises(IndexError):
        IdxRawSampleSequence(images, labels)[i]


def test_leer_la_misma_posicion_dos_veces_da_lo_mismo_aunque_haya_otras_lecturas_en_medio(tmp_path):
    # el fichero tiene un puntero de lectura; x[i] no puede depender de donde quedo
    images, labels, imagenes, etiquetas = _par_distinguible(tmp_path)
    sequence = IdxRawSampleSequence(images, labels)

    primera = sequence[2]
    sequence[0]
    segunda = sequence[2]

    assert primera == segunda == RawSample(x=imagenes[2], y_true=etiquetas[2])


# ---------- si algo falla al abrir, no queda ningun fichero abierto ----------

def _espiar_open(monkeypatch):
    """Sustituye el `open` del adaptador por un espia que deja pasar la llamada real.

    Devuelve la lista, que se va llenando, de los ficheros que el adaptador abrio.
    """
    abiertos = []

    def espia(*args, **kwargs):
        fichero = open(*args, **kwargs)  # aqui `open` es el de builtins: el parche es del adaptador
        abiertos.append(fichero)
        return fichero

    monkeypatch.setattr(adaptador, "open", espia, raising=False)
    return abiertos


def test_si_falla_la_cabecera_del_segundo_fichero_no_queda_ninguno_abierto(tmp_path, monkeypatch):
    images, labels = _par_valido(tmp_path)
    contenido = bytearray(labels.read_bytes())
    contenido[0] = 0xFF
    labels.write_bytes(contenido)
    abiertos = _espiar_open(monkeypatch)

    with pytest.raises(ValueError, match="magic"):
        IdxRawSampleSequence(images, labels)

    # 2 = el espia si ve los ficheros; sin esto, "todos cerrados" pasaria en vacio
    assert len(abiertos) == 2
    assert all(fichero.closed for fichero in abiertos)


def test_si_falla_tras_abrir_los_dos_ficheros_no_queda_ninguno_abierto(tmp_path, monkeypatch):
    # cada fichero es valido por separado: solo discrepan en N, y eso se ve con los dos abiertos
    images = write_idx(tmp_path, "images", dims=(3, 3, 40))
    labels = write_idx(tmp_path, "labels", dims=(2,))
    abiertos = _espiar_open(monkeypatch)

    with pytest.raises(ValueError, match="differs"):
        IdxRawSampleSequence(images, labels)

    assert len(abiertos) == 2
    assert all(fichero.closed for fichero in abiertos)


def test_si_el_segundo_fichero_no_existe_el_primero_queda_cerrado(tmp_path, monkeypatch):
    images, _ = _par_valido(tmp_path)
    abiertos = _espiar_open(monkeypatch)

    with pytest.raises(FileNotFoundError):
        IdxRawSampleSequence(images, tmp_path / "no_existe")

    # solo se llego a abrir el primero: la segunda apertura fue la que fallo
    assert len(abiertos) == 1
    assert abiertos[0].closed


# ---------- cierre con `with` ----------

def test_enter_devuelve_el_propio_objeto(tmp_path):
    images, labels = _par_valido(tmp_path)
    sequence = IdxRawSampleSequence(images, labels)

    with sequence as dentro:
        assert dentro is sequence


def test_al_salir_del_with_los_ficheros_quedan_cerrados(tmp_path, monkeypatch):
    images, labels = _par_valido(tmp_path)
    abiertos = _espiar_open(monkeypatch)

    with IdxRawSampleSequence(images, labels) as sequence:
        sequence[0]
        assert not any(fichero.closed for fichero in abiertos)

    assert len(abiertos) == 2
    assert all(fichero.closed for fichero in abiertos)


def test_los_ficheros_quedan_cerrados_aunque_el_cuerpo_del_with_lance(tmp_path, monkeypatch):
    images, labels = _par_valido(tmp_path)
    abiertos = _espiar_open(monkeypatch)

    with pytest.raises(RuntimeError, match="fallo del usuario"):
        with IdxRawSampleSequence(images, labels):
            raise RuntimeError("fallo del usuario")

    assert len(abiertos) == 2
    assert all(fichero.closed for fichero in abiertos)
