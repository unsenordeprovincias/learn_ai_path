---
title: Ruta de implementación: de perceptrones a grafo de operaciones
tags:
  - redes-neuronales
  - backpropagation
  - grafos
  - mnist
---

# Ruta de implementación: de perceptrones a grafo de operaciones

## Idea rectora

Construir primero una red densa y columnar, con responsabilidades claras, y generalizar después su ejecución a un grafo. No adelantar el grafo general antes de que XOR y el *backward* básico sean fiables.

```text
Fase 1: Vector → Perceptron → Layer → NeuralNet → XOR
Fase 2: Perceptron = WeightedSum + Activation
Fase 3: Node + Graph de operaciones → MNIST
```

Los pesos son estado entrenable; el caché es estado efímero de una pasada *forward*. La arquitectura debe poder serializarse sin ninguno de los dos, y el *checkpoint* entrenado puede incluir los pesos.

---

## Fase 1 — Red columnar mínima

### Topología

```text
Vector(n)
   ↓
Perceptron
   ↓
Layer(n → m)
   ↓
NeuralNet([Layer...])
```

- `Vector`: contenedor de valores y operaciones vectoriales elementales.
- `Perceptron`: pesos, sesgo, suma ponderada, activación y *backward* local.
- `Layer`: colección de perceptrones que transforma un vector en otro vector.
- `NeuralNet`: secuencia ordenada de capas.

Primer hito: aprender XOR con una red `2 → 2 → 1` y verificar que converge desde varias inicializaciones.

### Caché y backward

Cada `Perceptron` conserva el mínimo necesario para derivar su última ejecución:

- entrada `x`;
- preactivación `z = w·x + b`;
- salida `a = activation(z)`.

El *backward* está encapsulado en el perceptrón: recibe el gradiente respecto a `a`, calcula los gradientes de `w` y `b`, y devuelve el gradiente respecto a `x`. `Layer` reúne los gradientes de sus perceptrones sobre el vector de entrada; `NeuralNet` los propaga capa a capa en orden inverso.

**Regla:** el caché corresponde a una ejecución concreta. Debe reemplazarse en cada `forward` y consumirse antes de ejecutar otro `forward` si el entrenamiento es por muestra.

### JSON columnar

Arquitectura, sin pesos:

```json
{
  "type": "NeuralNet",
  "input_size": 2,
  "layers": [
    { "type": "Layer", "units": 2, "activation": "sigmoid" },
    { "type": "Layer", "units": 1, "activation": "sigmoid" }
  ]
}
```

Las dimensiones de entrada de cada capa se deducen de la capa anterior. En un *checkpoint* entrenado, cada unidad añade sus `weights` y `bias`; no añadir cachés ni gradientes acumulados.

---

## Fase 2 — Descomponer el perceptrón

### Topología

Mantener la API de `Perceptron`, pero hacerlo una composición explícita:

```text
Perceptron(x) = Activation(WeightedSum(x, w, b))
```

- `WeightedSum`: calcula `z = w·x + b`.
- `Activation`: calcula `a = f(z)`.
- `Perceptron`: orquesta ambos; sigue siendo la unidad utilizada por `Layer`.

Esto cambia la estructura interna, no la topología externa de la Fase 1. XOR debe seguir funcionando sin que `Layer` o `NeuralNet` conozcan la división.

### Caché y backward

El caché se separa por responsabilidad:

- `WeightedSum` guarda `x` (y usa sus parámetros `w`, `b`) para calcular `∂L/∂w`, `∂L/∂b` y `∂L/∂x`.
- `Activation` guarda `z` o `a`, según la derivada sea más directa, para calcular `∂L/∂z`.
- `Perceptron` sólo coordina: `backward` de activación y después `backward` de suma ponderada.

El resultado es idéntico al de la Fase 1, pero las reglas locales ya están aisladas. Este es el punto adecuado para mejorar activaciones, inicialización, optimizador o tratamiento por lotes sin reescribir la semántica de cada operación.

---

## Fase 3 — Grafo general de operaciones

### Modelo

Generalizar la ejecución: toda transformación es un `Node`, no sólo un perceptrón.

```text
Node
 ├─ forward(inputs) → output
 ├─ backward(gradient) → gradients_for_inputs
 └─ cache

Operaciones iniciales
 ├─ WeightedSum
 ├─ Sigmoid
 ├─ Add
 └─ Multiply

Graph
 ├─ nodes
 ├─ connections
 ├─ input nodes
 └─ output nodes
```

Un `Graph` ejecuta los nodos en orden topológico durante *forward* y en orden topológico inverso durante *backward*. Al principio debe ser un DAG: no hay ciclos ni estado recurrente.

`Perceptron` deja de ser una primitiva necesaria del grafo: se expresa como el subgrafo `WeightedSum → Sigmoid` (o cualquier activación). Puede conservarse como constructor de conveniencia para crear ese patrón.

### Caché y backward distribuido

Cada nodo guarda exclusivamente lo necesario para su derivada local. El grafo no interpreta las fórmulas de `Sigmoid`, `Add`, etc.; sólo entrega entradas en *forward* y enruta gradientes en *backward*.

En una convergencia, una salida alimenta a varios nodos. El gradiente de ese valor es la suma de todas sus contribuciones:

```text
             ┌→ rama A → ∂L/∂v desde A ┐
valor v ─────┤                         ├→ grad(v) = suma de contribuciones
             └→ rama B → ∂L/∂v desde B ┘
```

Por tanto, el grafo mantiene un acumulador de gradiente por salida/nodo durante cada *backward*. Un nodo sólo ejecuta su derivada local cuando ya ha recibido las contribuciones de todos sus consumidores. En las operaciones con varias entradas, `backward` devuelve un gradiente por cada arista de entrada; si una misma fuente aparece más de una vez, esas contribuciones también se suman.

Los gradientes de parámetros (`weights`, `bias`) se acumulan por parámetro para el lote y el optimizador los aplica una vez por actualización. El caché se limpia tras la actualización o al iniciar una nueva pasada para evitar mezclar ejecuciones.

### JSON de grafo de perceptrones

Paso intermedio: los nodos siguen siendo perceptrones, pero las conexiones ya definen la topología. El número de entradas se deduce de las aristas entrantes, no se repite en el nodo.

```json
{
  "type": "PerceptronGraph",
  "inputs": ["x1", "x2"],
  "nodes": [
    { "id": "h1", "type": "Perceptron", "activation": "sigmoid" },
    { "id": "h2", "type": "Perceptron", "activation": "sigmoid" },
    { "id": "y", "type": "Perceptron", "activation": "sigmoid" }
  ],
  "connections": [
    { "from": "x1", "to": "h1" },
    { "from": "x2", "to": "h1" },
    { "from": "x1", "to": "h2" },
    { "from": "x2", "to": "h2" },
    { "from": "h1", "to": "y" },
    { "from": "h2", "to": "y" }
  ],
  "outputs": ["y"]
}
```

### JSON de grafo general de operaciones

El grafo describe operaciones y sus dependencias. Los parámetros entrenables se referencian o se incluyen como estado de *checkpoint*, pero el caché no se serializa.

```json
{
  "type": "ComputationGraph",
  "inputs": ["x", "w", "b"],
  "nodes": [
    { "id": "wx", "type": "Multiply" },
    { "id": "z", "type": "Add" },
    { "id": "y", "type": "Sigmoid" }
  ],
  "connections": [
    { "from": "x", "to": "wx", "input": 0 },
    { "from": "w", "to": "wx", "input": 1 },
    { "from": "wx", "to": "z", "input": 0 },
    { "from": "b", "to": "z", "input": 1 },
    { "from": "z", "to": "y", "input": 0 }
  ],
  "outputs": ["y"]
}
```

El campo `input` fija el puerto de destino cuando el orden importa (`Multiply`, `Add`, `WeightedSum`). Para operaciones con entradas conmutativas puede seguir siendo explícito por consistencia.

---

## Objetivo experimental — MNIST + clase de rechazo

Entrenar una red para MNIST con 11 salidas:

- salidas `0` a `9`: dígitos MNIST;
- salida `10`: ruido blanco / entrada sin dígito, interpretada como `NaN` o rechazo.

### Preguntas a medir

1. Exactitud en MNIST limpio.
2. Tasa de detección del ruido blanco como clase 10.
3. Confianza máxima de las diez clases de dígitos ante ruido: debe disminuir cuando la salida 11 aumenta.
4. Generalización fuera del entrenamiento: ruido con varianzas, intensidades y distribuciones distintas; imágenes degradadas; mezclas de dígito y ruido.
5. Compromiso: cuánto mejora el rechazo sin degradar indebidamente la clasificación de dígitos válidos.

La clase 11 no demuestra por sí sola comprensión de “no sé”; mide una forma específica de generalización frente a la distribución de ruido elegida. Conviene separar siempre los generadores de ruido de entrenamiento, validación y prueba.

---

## Criterios para avanzar

- **Fase 1 → 2:** XOR converge, el gradiente se propaga correctamente y la arquitectura se guarda/carga.
- **Fase 2 → 3:** las reglas locales de suma ponderada y activación están verificadas de forma independiente.
- **Fase 3 → MNIST:** orden topológico, acumulación en convergencias, parámetros compartidos y *checkpoint* del grafo funcionan de forma determinista.

La prioridad es preservar invariantes: mismo *forward*, gradientes locales correctos, caché ligado a una pasada y suma correcta de gradientes donde el grafo se bifurca o converge.
