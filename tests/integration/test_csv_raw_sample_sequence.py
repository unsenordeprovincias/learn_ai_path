import pytest

import mnist.adapters.csv_raw_sample_sequence as adaptador
from mnist.adapters.csv_raw_sample_sequence import CsvRawSampleSequence
from mnist.domain.samples import RawSample
from tests.csv_files import csv_bytes, write_csv
from tests.open_spy import spy_on_open

# Dos ejemplos de forma (2, 4): cada fila del CSV tiene 1 etiqueta + 8 pixeles = 9 columnas.
IMAGES = [bytes(range(8)), bytes(range(8, 16))]
LABELS = [bytes([3]), bytes([7])]
X_SHAPE = (2, 4)


# ---------- validaciones al abrir ----------
# Cada test rompe UNA cosa de un CSV valido y comprueba que el error nombra esa cosa.

@pytest.mark.parametrize(
    "x_shape", [(2, 3), (3, 4)], ids=["forma menor que las filas", "forma mayor que las filas"]
)
def test_forma_incompatible_con_el_ancho_de_las_filas_se_rechaza(tmp_path, x_shape):
    path = write_csv(tmp_path, "data.csv", IMAGES, LABELS)

    with pytest.raises(ValueError, match="columns"):
        CsvRawSampleSequence(path, x_shape)


def test_una_fila_con_otro_numero_de_columnas_se_rechaza_y_el_error_dice_cual(tmp_path):
    # la forma es la correcta para las filas 1 y 3; la fila 2 tiene un pixel de menos
    images = [bytes(range(8)), bytes(range(7)), bytes(range(8))]
    path = write_csv(tmp_path, "data.csv", images, [bytes([1]), bytes([2]), bytes([3])])

    with pytest.raises(ValueError, match="line 2 has 8 columns"):
        CsvRawSampleSequence(path, X_SHAPE)


def test_una_linea_en_blanco_en_medio_se_rechaza(tmp_path):
    filas = csv_bytes(IMAGES, LABELS).split(b"\n")  # [fila1, fila2, b""]
    path = tmp_path / "data.csv"
    path.write_bytes(filas[0] + b"\n\n" + filas[1] + b"\n")

    with pytest.raises(ValueError, match="blank line 2"):
        CsvRawSampleSequence(path, X_SHAPE)


@pytest.mark.parametrize("final", [b"\n", b"\n\n\n", b"\r\n\r\n"], ids=["salto final normal", "varias en blanco", "en blanco con CRLF"])
def test_lineas_en_blanco_al_final_se_toleran_y_no_cuentan_como_ejemplos(tmp_path, final):
    path = tmp_path / "data.csv"
    path.write_bytes(csv_bytes(IMAGES, LABELS).rstrip(b"\n") + final)

    with CsvRawSampleSequence(path, X_SHAPE) as sequence:
        assert len(sequence) == 2


@pytest.mark.parametrize("contenido", [b"", b"\n\n"], ids=["vacio", "solo lineas en blanco"])
def test_un_fichero_sin_ejemplos_se_rechaza(tmp_path, contenido):
    # un CSV vacio suele ser una descarga fallida, no un conjunto de datos de cero ejemplos
    path = tmp_path / "data.csv"
    path.write_bytes(contenido)

    with pytest.raises(ValueError, match="no samples"):
        CsvRawSampleSequence(path, X_SHAPE)


# ---------- el contenido de las celdas se valida al leer, no al abrir ----------

@pytest.mark.parametrize(
    "celda",
    [b"", b"abc", b"256", b"-1"],
    ids=["vacia", "no numerica", "mayor que 255", "negativa"],
)
@pytest.mark.parametrize("columna", [0, 3], ids=["etiqueta", "pixel"])
def test_una_celda_invalida_solo_falla_al_leer_esa_fila(tmp_path, celda, columna):
    filas = csv_bytes(IMAGES, LABELS).splitlines()
    campos = filas[1].split(b",")
    campos[columna] = celda
    filas[1] = b",".join(campos)
    path = tmp_path / "data.csv"
    path.write_bytes(b"\n".join(filas) + b"\n")

    with CsvRawSampleSequence(path, X_SHAPE) as sequence:
        assert len(sequence) == 2  # abrir no mira el contenido de las celdas
        assert sequence[0] == RawSample(x=IMAGES[0], y_true=LABELS[0])  # las buenas se leen
        with pytest.raises(ValueError):
            sequence[1]


def test_las_filas_con_fin_de_linea_windows_se_leen_igual(tmp_path):
    path = write_csv(tmp_path, "data.csv", IMAGES, LABELS, newline=b"\r\n")

    with CsvRawSampleSequence(path, X_SHAPE) as sequence:
        assert len(sequence) == 2
        assert [sequence[0], sequence[1]] == [
            RawSample(x=IMAGES[0], y_true=LABELS[0]),
            RawSample(x=IMAGES[1], y_true=LABELS[1]),
        ]


# ---------- recursos: si algo falla no queda nada abierto, y `with` cierra ----------

def test_si_falla_al_abrir_el_fichero_queda_cerrado(tmp_path, monkeypatch):
    path = write_csv(tmp_path, "data.csv", IMAGES, LABELS)
    abiertos = spy_on_open(monkeypatch, adaptador)

    with pytest.raises(ValueError, match="columns"):
        CsvRawSampleSequence(path, (2, 3))

    # 1 = el espia si ve el fichero; sin esto, "todos cerrados" pasaria en vacio
    assert len(abiertos) == 1
    assert all(fichero.closed for fichero in abiertos)


def test_enter_devuelve_el_propio_objeto_y_al_salir_del_with_el_fichero_queda_cerrado(
    tmp_path, monkeypatch
):
    path = write_csv(tmp_path, "data.csv", IMAGES, LABELS)
    abiertos = spy_on_open(monkeypatch, adaptador)
    sequence = CsvRawSampleSequence(path, X_SHAPE)

    with sequence as dentro:
        assert dentro is sequence
        assert not any(fichero.closed for fichero in abiertos)

    assert len(abiertos) == 1
    assert all(fichero.closed for fichero in abiertos)
