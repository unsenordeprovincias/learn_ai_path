# Lectura de datos MNIST

## La idea: acceso por posición, sin transformar

Los datos entran al programa por un único contrato, `RawSampleSequence`
([ports/raw_sample_sequence.py](ports/raw_sample_sequence.py)): un objeto de solo lectura
con `len(secuencia)`, la forma de un ejemplo de entrada (`secuencia.x_shape`, p. ej.
`(28, 28)`) y `secuencia[i]`, que devuelve un `RawSample(x, y_true)`.

Tres decisiones explican su forma:

- **Por posición y sin estado de recorrido.** Leer `secuencia[i]` no cambia lo que devolverá la
  siguiente lectura. Quien quiera barajar elige los índices; quien quiera otra época vuelve
  a recorrer. La secuencia no sabe de épocas ni de orden.
- **`IndexError` para todo lo que no sea `0 <= i < n`, negativos incluidos.** Es una
  desviación deliberada de las secuencias de Python, donde `secuencia[-1]` es el último: aquí un
  `-1` casi siempre es un error de cálculo de índices y preferimos que falle a leer en
  silencio el último ejemplo. (En el adaptador IDX, sin esta comprobación, un `-1` leía
  bytes de la cabecera como si fueran una imagen.)
- **"Raw" = sin transformar para el modelo.** `x` e `y_true` son los bytes tal como están
  guardados. Dividir entre 255, pasar la etiqueta a one-hot o decidir cuántas clases hay
  son decisiones del modelo, no del formato de fichero, y las toma quien consume la
  secuencia. El resultado transformado es un `NetSample`, ambos en
  [domain/samples.py](domain/samples.py) e inmutables.

La forma de la entrada es propiedad del dato, no del formato. IDX la trae en su cabecera;
un CSV no, y por eso su adaptador la recibe de quien lo construye.

## Adaptadores

| Adaptador | Formato | De dónde sale `x_shape` | Cómo llega a `secuencia[i]` |
|---|---|---|---|
| `IdxRawSampleSequence` | IDX descomprimido, dos ficheros (imágenes y etiquetas) | de la cabecera | `seek` a `cabecera + i * tamaño_de_registro`: los registros miden lo mismo |
| `CsvRawSampleSequence` | CSV sin cabecera, etiqueta en la primera columna | parámetro del constructor, validado contra el ancho de cada fila | índice de desplazamientos construido al abrir: las líneas miden distinto |

Los dos son recursos con cierre explícito (`with`). Si algo falla al abrir no queda ningún
fichero abierto. El cierre es cosa de los adaptadores que poseen ficheros, no del port.

Un `.gz` no se lee: falla por magic incorrecto (`0x1f8b0808`). El acceso por posición
exige un fichero descomprimido, porque en un flujo gzip no se puede saltar al byte
`cabecera + i * tamaño` sin descomprimir lo anterior.

Las rutas de los ficheros reales están en `config/settings.py` y se sobrescriben con
variables de entorno (p. ej. `TRAIN_IMAGES_PATH`).

## Cómo se comprueba que el port no depende de la infraestructura

- **Batería de contrato** (`tests/contract`): los mismos tests para IDX, CSV y un
  adaptador en memoria sin ficheros, que hace de doble de test. Un adaptador nuevo se
  registra en una tabla y pasa exactamente los mismos casos.
- **Fronteras de importación** (`tests/architecture`): `domain` y `ports` solo pueden
  importar lo que una lista de permitidos autoriza. Es una lista de permitidos y no de
  prohibidos porque esta última no detecta lo que no se nos ocurrió prohibir.
- **Ficheros reales** (`tests/integration/test_idx_real_files.py`): los tests sintéticos
  usan un helper y un adaptador escritos por la misma persona, así que un malentendido
  común del formato los pasaría. Estos comparan contra literales verificados por fuera.

## El camino antiguo, retirado

Antes existía `SampleSource`, un port con `load() -> Iterator[NetSample]`, y su adaptador
`IdxMnistSource`, que recibía los ficheros ya leídos como `bytes`. Se retiró porque
combinaba tres cosas que aquí están separadas:

- **Iteración de un solo uso.** Sin `len`, sin acceso por índice y sin barajar, quien
  necesitara alguna tenía que materializar todo con `list(source.load())`, es decir, 60.000
  `Vector` de 784 flotantes de Python en memoria (coste no medido).
- **Fichero entero en memoria** como `bytes`, en lugar de leer cada registro por posición.
- **Decisiones de modelo dentro del adaptador:** dividía entre 255 y hacía el one-hot con 10
  clases fijas.

Adónde fue cada comportamiento:

| Antes | Ahora |
|---|---|
| `parse_idx_header` y comprobaciones de tamaño y de N | validaciones al abrir de `IdxRawSampleSequence`; el tamaño real se contrasta con la cabecera *antes* de fiarse de las dimensiones |
| Normalizar entre 0 y 1 y one-hot con 10 clases | pendiente del consumidor de la secuencia (el muestreador, fuera de esta etapa), que produce `NetSample` |
| Recorrer con `load()` una vez por época | recorrer índices sobre la secuencia |
| Fuente a partir de `bytes` en memoria | el adaptador en memoria de los tests cumple ese papel, solo para pruebas |

Los tests del contrato de `load()` (iterador nuevo en cada llamada, iterador que se agota,
`islice`) no tienen equivalente a propósito: describen la semántica de iteración que este
diseño elimina.

`NetSample` se conserva: lo usan los playgrounds (`playgrounds/xor`, `playgrounds/slicer`).
Ningún código de `src` lo produce todavía.

El código retirado sigue disponible en el commit `8ef3f0f`:
`git show 8ef3f0f:src/mnist/adapters/idx_mnist_source.py`.

## Límites conocidos

- Si un fichero cambia de tamaño después de abrirlo, los adaptadores no lo detectan.
- El CSV acepta `1_0`, ` 5` y `+5` como enteros (los acepta `int()`), y rechaza `255.0`. El
  contenido de sus celdas se valida al leer `secuencia[i]`, no al abrir.
- Los tests con ficheros IDX reales contrastan cabecera, tamaño y etiquetas contra literales
  externos. Que la imagen `i` sea de verdad el dígito de la etiqueta `i` se comprobó una
  sola vez con una prueba desechable (que no está en la suite), leyendo los ficheros con
  cortes directos de bytes y no con el adaptador: las tres primeras imágenes dibujadas eran
  un 5, un 0 y un 4, y un clasificador por centroide acertó el 77,4 % con las etiquetas
  alineadas frente al 13,8 % con las etiquetas desplazadas una posición.
