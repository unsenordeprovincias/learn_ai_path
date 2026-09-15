from mnist.domain.functions import mse, sigmoid
from mnist.domain.models import NeuralNet, Layer, Vector, Sample
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

l1 = Layer(2, 1, sigmoid)
nn = NeuralNet([l1], mse)

dataset = [
        Sample(x=Vector[0, 0], y_true=Vector[0]),
        Sample(x=Vector[0, 1], y_true=Vector[0]),
        Sample(x=Vector[1, 0], y_true=Vector[0]),
        Sample(x=Vector[1, 1], y_true=Vector[1])
    ]

def sigmoid_numpy(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def fCorte(W1: np.ndarray, W2: np.ndarray, x1: float, x2: float,
           k: float, factivacion) -> np.ndarray:
    """Salida del perceptron y=factivacion(x1*w1 + x2*w2 + k) evaluada
    sobre toda la malla (w1, w2) a la vez."""
    return factivacion(x1 * W1 + x2 * W2 + k)

def calcular_sabanas(dataset: list[Sample], b: float,
                      rango_min: float, rango_max: float,
                      n: int = 80) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    """Devuelve (w1_vals, w2_vals, sabanas_individuales, sabana_promedio)."""
    w1_vals = np.linspace(rango_min, rango_max, n)
    w2_vals = np.linspace(rango_min, rango_max, n)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    sabanas = []
    for sample in dataset:
        x1, x2 = sample.x[0], sample.x[1]
        y_true = sample.y_true[0]
        Y_pred = fCorte(W1, W2, x1, x2, b, sigmoid_numpy)
        Z = 0.5 * (y_true - Y_pred) ** 2
        sabanas.append(Z)

    Z_promedio = sum(sabanas) / len(sabanas)
    return w1_vals, w2_vals, sabanas, Z_promedio

def figura_cortes_superpuestos(dataset: list[Sample], b: float,
                                rango_min: float = -6, rango_max: float = 18,
                                n: int = 80) -> go.Figure:
    """Los 4 cortes por dato + el promedio, superpuestos en el mismo (w1, w2, perdida)."""
    w1_vals, w2_vals, sabanas, Z_promedio = calcular_sabanas(
        dataset, b, rango_min, rango_max, n
    )

    nombres = [f"({int(s.x[0])},{int(s.x[1])}) -> {int(s.y_true[0])}" for s in dataset]
    nombres.append("Promedio")
    todas = sabanas + [Z_promedio]
    colores = sample_colorscale('Turbo', np.linspace(0, 1, len(todas)))

    fig = go.Figure()
    for Z, nombre, color in zip(todas, nombres, colores):
        fig.add_trace(go.Surface(
            x=w1_vals, y=w2_vals, z=Z,
            name=nombre,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=0.55,
            showlegend=True
        ))

    n_traces = len(todas)
    botones = [dict(label='Todas', method='update', args=[{'visible': [True] * n_traces}])]
    for i, nombre in enumerate(nombres):
        vis = [False] * n_traces
        vis[i] = True
        botones.append(dict(label=nombre, method='update', args=[{'visible': vis}]))

    fig.update_layout(
        title=f"AND — 4 cortes + promedio, b = {b}",
        scene=dict(xaxis_title='w1', yaxis_title='w2', zaxis_title='pérdida'),
        updatemenus=[dict(buttons=botones, direction='down', x=1.15, y=1)],
        legend=dict(x=1.15, y=0.7),
        height=800
    )
    return fig


if __name__ == '__main__':
    import webview

    B = -12.0  # el bias real al que converge el entrenamiento de AND

    fig = figura_cortes_superpuestos(dataset, b=B)
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')

    webview.create_window(f'Slicer AND — b={B}', html=html)
    webview.start()