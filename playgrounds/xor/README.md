# Playground: XOR con perceptrones

## Objetivo

Conseguir que una red construida a partir del `Perceptron` que ya existe en
`src/mnist/domain/models.py` aprenda la función XOR:

| x1 | x2 | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

## Por qué XOR

Es el ejemplo clásico para mostrar los límites de un perceptrón simple: XOR no
es linealmente separable, así que un único perceptrón (una sola frontera de
decisión lineal) no puede resolverlo por más que se entrene. Sirve para
justificar, de forma tangible, el salto de "un perceptrón" a "una red de
perceptrones" (al menos una capa oculta).

## Punto de partida

Ya disponibles en el dominio (`src/mnist/domain/models.py`):

- `Vector`: suma, producto escalar, producto vectorial (`@`), inicialización
  aleatoria (`Vector.initialize`).
- `Perceptron`: pesos + bias inicializados al azar, `weighted_sum`, `output`
  (aplica la función de activación) y `correct` (ajusta pesos y bias con un
  delta).
- `Layer`: agrupa varios `Perceptron` (mismo nº de entradas, misma función de
  activación por defecto) y expone su propio `output`, que devuelve un
  `Vector` con la salida de cada perceptrón de la capa.

Este playground es un consumidor de ese dominio, no debería necesitar
modificarlo salvo que se descubra que falta algo imprescindible.

## Ruta de aprendizaje

### Etapa 1 — La red, chapuceramente, sin aprendizaje

Construir a mano (sin generalizar, sin bucles de entrenamiento, sin buscar
elegancia) esta topología concreta:

- Un vector de entrada (`x1`, `x2`).
- Dos perceptrones en una capa oculta, cada uno recibe el vector de entrada
  completo.
- Un tercer perceptrón que recibe la síntesis (las dos salidas) de los
  anteriores y produce la señal de salida final.

El único objetivo de esta etapa es que la señal fluya de principio a fin
(forward pass manual de la red) usando el `Perceptron` que ya existe. Nada de
aprendizaje todavía: los pesos se fijan a mano, ya calculados para que la red
acierte XOR desde el principio (activación escalón):

```
Capa oculta:
  Neurona 1: w=[1, 1], bias=-0.5   → aprende OR
  Neurona 2: w=[1, 1], bias=-1.5   → aprende AND

Capa salida:
  Neurona: w=[1, -2], bias=-0.5    → combina OR y NOT(AND)
```

Es decir, XOR = OR AND NOT(AND) — la capa oculta calcula esas dos señales por
separado y la capa de salida las combina.

#### Solución implementada

En `playgrounds/xor/__init__.py`, la red son dos `Layer` (2 entradas → 2
perceptrones la oculta, 2 entradas → 1 perceptrón la de salida), con los
pesos y bias de la tabla anterior asignados a mano tras crearlas. `layer1` y
`layer2` son variables de módulo: se construyen una sola vez, al importar el
módulo, y `XOR` es una función normal que las reutiliza en cada llamada:

```python
layer1 = Layer(2, 2, _step)
layer1[0].weights = Vector([1, 1])
...

def XOR(x1: int, x2: int) -> float:
    v = layer1.output(Vector([x1, x2]))
    r = layer2.output(v)
    return r[0]
```

Esto ya satisface tal cual el test que se escribió primero
(`tests/playgrounds/xor/test_xor.py`), que llama a `XOR(1, 1)` directamente
como si fuera una función:

```python
from playgrounds.xor import XOR

def test_XOR():
    assert XOR(1, 1) == 0
    ...
```

La primera versión de esta solución resolvía lo mismo con una clase y una
metaclase (`XOR` como clase "no instanciable" cuya llamada, vía
`metaclass.__call__`, ejecutaba la red). Funcionaba, pero era más maquinaria
de la que el problema pedía: el módulo ya es un singleton en Python, así que
no hace falta fingir uno con una clase para conseguir "construir la red una
sola vez". Se descartó a favor de esta versión más simple.

Pero ¿qué sentido tiene esto? AND, OR y XOR ya sabemos calcularlos a mano.
Es una buena simulación de neuronas, pero ¿qué aporta? Si en lugar de pesos y
bias diseñados, la red fuera capaz de encontrarlos por sí misma, eso sería
una forma de aprendizaje.

### Etapa 2 — ¿Puede aprender? ¿Cómo?

Con la red de la etapa 1 ya construida y entendida, abrir la pregunta de
cómo entrenarla:

- Cómo se calcula el error de la red frente al resultado esperado (XOR).
- Cómo se traduce ese error en los `delta_weights` / `delta_bias` que espera
  `Perceptron.correct` para cada uno de los tres perceptrones (esto es, en
  esencia, backpropagation a mano).
- Criterio de parada del entrenamiento (nº de épocas, error mínimo, etc.).

## Fuera de alcance por ahora

Nada de numpy, nada de librerías de ML. El objetivo pedagógico sigue siendo
entender cada pieza construyéndola desde cero, igual que con `Vector` y
`Perceptron`.
