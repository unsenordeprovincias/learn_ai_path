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

#### Solución implementada

`NeuralNet.backward` (en `src/mnist/domain/models.py`) calcula y aplica el
gradiente de cada perceptrón, capa por capa, de atrás hacia adelante.

**El error de cada neurona se diagnostica por separado, como un escalar, no
como una operación vectorizada sobre toda la capa.** Para cada perceptrón se
calcula su propio `activation_sensivity` (cuánto responde su salida ante un
cambio en su propio weighted_sum) y su propio `local_error` (cuánta culpa
tiene ese weighted_sum concreto en el error total), como dos números sueltos,
no como componentes de un vector que se calcula de una vez para toda la
capa. Esto no es una cuestión de rendimiento — es reflejar fielmente qué es
backpropagation en realidad: la regla de la cadena aplicada nodo por nodo.
Cada neurona solo necesita, para diagnosticarse a sí misma, dos cosas propias
— la parte del error que le corresponde a ella y su propia pendiente de
activación — y no le hace falta saber nada de las demás neuronas de su capa
para hacerlo. Calcularlo así, uno a uno, deja ver esa independencia; hacerlo
de golpe con vectores (aunque dé el mismo resultado y sea más eficiente)
esconde el hecho de que en el fondo son N cálculos idénticos e
independientes, no un único cálculo "de capa".

**Propagar el error hacia la capa anterior, en cambio, sí necesita la capa
entera a la vez.** Ahí la pregunta cambia: ya no es "¿cuánto le importa a
esta neurona su propio error?" sino "¿cuánta culpa tiene cada entrada de la
capa, sumando lo que aportó a *todas* las neuronas que la usaron?". Repartir
esa culpa exige combinar, para cada entrada, el error de cada neurona con el
peso que esa neurona le dio a esa entrada — una suma que cruza neuronas, y
por eso ahí sí hace falta trabajar con la capa entera como vector y matriz,
no perceptrón a perceptrón.

Esa combinación es un producto matriz-vector, pero con los pesos organizados
al revés de como se guardan de forma natural. Los pesos de una capa vienen
"una fila por neurona" (`perceptron.cache.weights` de cada perceptrón, que es
justo lo que hace falta para el forward pass de esa neurona). Para repartir
el error hacia atrás hace falta leer esa misma tabla "una fila por entrada"
— la traspuesta —, porque la culpa de una entrada concreta se arma juntando
un pedacito de cada neurona, no todos los pedacitos de una sola:

```python
W_transposed = Matrix([perceptron.cache.weights for perceptron in layer]).T
loss_grad = W_transposed @ local_error
```

**El gradiente analítico se contrasta contra uno calculado de forma
independiente.** `tests/conftest.py` expone `numerical_gradient`, que estima
la derivada de cada peso y cada bias por diferencia finita — perturbando ese
único parámetro un poquito hacia cada lado y viendo cuánto cambia la
pérdida, sin usar ninguna fórmula de backprop. Es una forma de verificación
habitual en redes neuronales: si el cálculo "manual" (la regla de la cadena
implementada a mano) y el cálculo "a ciegas" (por diferencias finitas)
coinciden, es una buena señal de que la implementación de `backward` es
correcta, no solo de que "compila". Para que la comparación sea directa,
`numerical_gradient` devuelve, por neurona, un `CachedGradient` — el mismo
tipo que ya usa el gradiente analítico — de forma que se comparan campo a
campo (`weights` contra `weights`, `bias` contra `bias`) sin necesidad de
traducir nada entre los dos cálculos.

## Fuera de alcance por ahora

Nada de numpy, nada de librerías de ML. El objetivo pedagógico sigue siendo
entender cada pieza construyéndola desde cero, igual que con `Vector` y
`Perceptron`.
