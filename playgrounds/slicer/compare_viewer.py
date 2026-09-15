"""
Slicer interactivo de AND: paisaje de perdida de un perceptron (2 entradas,
1 salida, sigmoide) para un rango amplio de b, con dos desplegables
independientes que se combinan:
  - "b": que corte del glaciar quieres ver
  - "etiqueta": todas las curvas de ese b, o solo una (los 4 datos o el promedio)

Los updatemenus nativos de plotly no se combinan entre si (cada uno fija un
estado absoluto), asi que se inyecta un <script> propio que escucha los dos
<select> y llama a Plotly.restyle con la interseccion de ambos filtros.
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from dataclasses import dataclass
import webview

from mnist.domain.models import Vector


@dataclass
class Sample:
    x: Vector
    y_true: Vector


def sigmoid_numpy(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fAND(W1: np.ndarray, W2: np.ndarray, x1: float, x2: float,
           k: float, factivacion) -> np.ndarray:
    return factivacion(x1 * W1 + x2 * W2 + k)


def dataset_and() -> list[Sample]:
    return [
        Sample(x=Vector[0, 0], y_true=Vector[0]),
        Sample(x=Vector[0, 1], y_true=Vector[0]),
        Sample(x=Vector[1, 0], y_true=Vector[0]),
        Sample(x=Vector[1, 1], y_true=Vector[1]),
    ]


def construir_figura(
    dataset: list[Sample],
    b_min: float = -15, b_max: float = 5, b_paso: float = 1,
    rango_min: float = -30, rango_max: float = 30,
    n: int = 50,
) -> tuple[go.Figure, list[float], list[str]]:

    w1_vals = np.linspace(rango_min, rango_max, n)
    w2_vals = np.linspace(rango_min, rango_max, n)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    etiquetas = [f"({int(s.x[0])},{int(s.x[1])})->{int(s.y_true[0])}" for s in dataset]
    etiquetas.append("Promedio")

    b_values = np.round(np.arange(b_max, b_min - b_paso / 2, -b_paso), 2)
    colores = sample_colorscale('Turbo', np.linspace(0, 1, len(etiquetas)))

    fig = go.Figure()
    b_de_cada_traza = []
    etiqueta_de_cada_traza = []

    for b in b_values:
        sabanas = []
        for sample in dataset:
            x1, x2 = sample.x[0], sample.x[1]
            y_true = sample.y_true[0]
            Y_pred = fAND(W1, W2, x1, x2, b, sigmoid_numpy)
            Z = 0.5 * (y_true - Y_pred) ** 2
            sabanas.append(Z)
        Z_promedio = sum(sabanas) / len(sabanas)
        todas_las_Z = sabanas + [Z_promedio]

        for Z, etiqueta, color in zip(todas_las_Z, etiquetas, colores):
            fig.add_trace(go.Surface(
                x=w1_vals, y=w2_vals, z=Z,
                name=etiqueta,
                colorscale=[[0, color], [1, color]],
                showscale=False,
                opacity=0.55,
                showlegend=True,
                visible=False,  # el JS decide la visibilidad inicial
            ))
            b_de_cada_traza.append(float(b))
            etiqueta_de_cada_traza.append(etiqueta)

    fig.update_layout(
        title="AND — cortes por b y por dato de entrada",
        scene=dict(xaxis_title='w1', yaxis_title='w2', zaxis_title='pérdida'),
        height=800,
        showlegend=True,
    )
    return fig, b_de_cada_traza, etiqueta_de_cada_traza


def generar_html(fig: go.Figure, b_de_cada_traza: list[float],
                  etiqueta_de_cada_traza: list[str],
                  b_inicial: float, ruta_salida: str) -> None:

    div_id = "grafico_slicer"
    cuerpo_plotly = fig.to_html(
        full_html=False, include_plotlyjs='cdn', div_id=div_id
    )

    b_values_unicos = sorted(set(b_de_cada_traza), reverse=True)
    etiquetas_unicas = list(dict.fromkeys(etiqueta_de_cada_traza))

    opciones_b = "\n".join(
        f'<option value="{b}" {"selected" if b == b_inicial else ""}>b = {b}</option>'
        for b in b_values_unicos
    )
    opciones_etiqueta = '<option value="__todas__">Todas</option>\n' + "\n".join(
        f'<option value="{e}">{e}</option>' for e in etiquetas_unicas
    )

    html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: sans-serif;">
  <div style="margin-bottom: 10px;">
    <label>b: <select id="selector_b">{opciones_b}</select></label>
    <label style="margin-left: 20px;">Etiqueta: <select id="selector_etiqueta">{opciones_etiqueta}</select></label>
  </div>

  {cuerpo_plotly}

  <script>
    const bDeCadaTraza = {json.dumps(b_de_cada_traza)};
    const etiquetaDeCadaTraza = {json.dumps(etiqueta_de_cada_traza)};
    const graficoDiv = document.getElementById("{div_id}");

    function actualizarVisibilidad() {{
      const bSeleccionado = parseFloat(document.getElementById("selector_b").value);
      const etiquetaSeleccionada = document.getElementById("selector_etiqueta").value;

      const visibilidad = bDeCadaTraza.map((b, i) => {{
        const coincideB = Math.abs(b - bSeleccionado) < 1e-6;
        const coincideEtiqueta = (etiquetaSeleccionada === "__todas__") ||
                                  (etiquetaDeCadaTraza[i] === etiquetaSeleccionada);
        return coincideB && coincideEtiqueta;
      }});

      Plotly.restyle(graficoDiv, {{visible: visibilidad}});
    }}

    document.getElementById("selector_b").addEventListener("change", actualizarVisibilidad);
    document.getElementById("selector_etiqueta").addEventListener("change", actualizarVisibilidad);

    actualizarVisibilidad();
  </script>
</body>
</html>
"""
    with open(ruta_salida, "w") as f:
        f.write(html)

if __name__ == '__main__':
    import webview
    import os

    fig, b_traza, etiqueta_traza = construir_figura(dataset_and())
    ruta_salida = "slicer_and_interactivo.html"
    generar_html(fig, b_traza, etiqueta_traza, b_inicial=-12.0, ruta_salida=ruta_salida)

    ruta_absoluta = os.path.abspath(ruta_salida)
    webview.create_window("Slicer AND interactivo", url=f"file://{ruta_absoluta}")
    webview.start()