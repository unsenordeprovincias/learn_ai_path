# Leer MNIST desde cero

*Bitácora de diseño de la capa de lectura de datos: un contrato, tres adaptadores y lo que
aprendimos comprobándolos.*

> **Estado a 2026-09-20.** 167 tests en verde y mypy sin errores. Unas 270 líneas de código
> de producción (contrato, tipos y dos adaptadores) respaldadas por más de un centenar de tests
> nuevos: se pasó de 78 a 167 tras retirar 33 del camino antiguo. Todo el código es Python
> puro, sin numpy.

## Índice

1. [En una página](#1-en-una-página)
2. [El punto de partida](#2-el-punto-de-partida)
3. [El mapa](#3-el-mapa)
4. [La idea central](#4-la-idea-central)
5. [El formato IDX en cinco minutos](#5-el-formato-idx-en-cinco-minutos)
6. [CSV: lo que IDX daba gratis](#6-csv-lo-que-idx-daba-gratis)
7. [Cómo trabajamos](#7-cómo-trabajamos)
8. [La bitácora, paso a paso](#8-la-bitácora-paso-a-paso)
9. [Trampas y sorpresas](#9-trampas-y-sorpresas)
10. [Decisiones y alternativas descartadas](#10-decisiones-y-alternativas-descartadas)
11. [Qué está verificado y qué es hipótesis](#11-qué-está-verificado-y-qué-es-hipótesis)
12. [Pendiente](#12-pendiente)
13. [Cómo convertirlo en una ruta didáctica](#13-cómo-convertirlo-en-una-ruta-didáctica)
14. [Cómo ejecutarlo](#14-cómo-ejecutarlo)
15. [Glosario](#15-glosario)

---

## 1. En una página

Los datos de MNIST entran al programa por **un único contrato**, `RawSampleSequence`: un objeto
de solo lectura con `len`, la forma de un ejemplo y acceso por posición.

```python
with IdxRawSampleSequence(images_path, labels_path) as secuencia:
    len(secuencia)       # 60000
    secuencia.x_shape    # (28, 28)
    secuencia[0]         # RawSample(x=<784 bytes>, y_true=b'\x05')
```

Tres ideas sostienen el diseño:

- **Por posición, sin estado.** No hay iterador ni "siguiente". Quien quiera barajar elige los
  índices.
- **Sin transformar ("raw").** Se devuelven los bytes tal como están guardados. Normalizar o
  pasar la etiqueta a one-hot es decisión del modelo, no del formato.
- **El formato es un detalle.** IDX, CSV o datos generados cumplen el mismo contrato y pasan
  exactamente los mismos tests.

Además de la lectura, esta bitácora recoge **cómo** se construyó (test primero, en rojo, y
comprobando que cada test puede fallar), qué salió mal por el camino y qué queda sin verificar.
Está pensada como material para escribir, algún día, una ruta didáctica sobre el tema.

## 2. El punto de partida

El proyecto es una red neuronal escrita desde cero: Python puro, sin numpy, arquitectura
hexagonal (dominio, puertos y adaptadores) y `uv` como gestor. Ya existían `Vector`,
`Perceptron`, `Layer` y `NeuralNet`. Faltaba lo que hay debajo de todo entrenamiento: **leer
los datos**.

**Alcance de esta etapa:** solo lectura de datos. Quedaron fuera el muestreo, el barajado, el
entrenamiento y las mediciones de rendimiento.

**Las fases del autor** determinan el diseño:

```text
IDX (los ficheros originales) → CSVs que se encuentren → CSVs generados a mano
        → eficiencia → experimentos propios
```

Si el formato va a cambiar tres veces y los datos van a fabricarse, atarse a IDX habría sido un
error. De ahí el contrato independiente del formato y la insistencia en comprobar esa
independencia (sección 4).

## 3. El mapa

```text
                  ┌────────────────────────────────────────────┐
  quien consume ─▶│  RawSampleSequence  (port)                 │
 (muestreador,    │  len · x_shape · secuencia[i] → RawSample  │
  herramientas)   └───────────────────▲────────────────────────┘
                                      │ lo cumplen
          ┌───────────────────────────┼───────────────────────────┐
 IdxRawSampleSequence        CsvRawSampleSequence        InMemoryRawSampleSequence
 seek + read                 índice de desplazamientos   doble de test
          │                            │
    ficheros IDX                 ficheros CSV
```

**Regla de dependencias:** `domain` no importa nada del proyecto salvo `domain`; `ports` solo
importa `domain`; los adaptadores dependen de `domain` y son la única capa que toca ficheros. Un
test comprueba las dos primeras (sección 8, paso 6b).

| Pieza | Fichero | Qué hace |
|---|---|---|
| `RawSample`, `NetSample` | [domain/samples.py](domain/samples.py) | El par `(x, y_true)`: en bruto y ya transformado. Inmutables. |
| `RawSampleSequence` | [ports/raw_sample_sequence.py](ports/raw_sample_sequence.py) | El contrato, con su semántica en los docstrings. |
| `IdxRawSampleSequence` | [adapters/idx_raw_sample_sequence.py](adapters/idx_raw_sample_sequence.py) | Lee IDX descomprimido por posición. |
| `CsvRawSampleSequence` | [adapters/csv_raw_sample_sequence.py](adapters/csv_raw_sample_sequence.py) | Lee CSV por posición con un índice de líneas. |
| `Settings` | [config/settings.py](config/settings.py) | Rutas de los ficheros reales, sobrescribibles por entorno. |
| Herramientas de test | `tests/idx_files.py`, `csv_files.py`, `open_spy.py`, `import_rules.py`, `in_memory_raw_sample_sequence.py` | Ficheros sintéticos, espía de `open`, comprobador de imports y doble en memoria. |

## 4. La idea central

### Por posición y sin estado de recorrido

Leer `secuencia[i]` no cambia lo que devolverá la siguiente lectura. La secuencia no conoce
épocas ni orden. Esto sirve a todo lo que viene después: barajar es elegir índices, y una época
nueva es volver a recorrer.

### `IndexError` para todo lo que no sea `0 <= i < n`, negativos incluidos

Es una **desviación deliberada** de las secuencias de Python, donde `x[-1]` es el último. Aquí
un `-1` casi siempre es un error de cálculo de índices, y preferimos que falle a leer en
silencio el último ejemplo.

No es un capricho teórico. Las tres primeras implementaciones ingenuas cayeron con esto (véase
la sección 9): cualquier lista de Python acepta negativos, así que si no lo dices en el
contrato y lo pruebas, cada adaptador lo resolverá a su manera.

### "Raw" = sin transformar para el modelo

`RawSample` lleva `x` e `y_true` como `bytes`. `NetSample` lleva los mismos campos ya como
`Vector`. El orden del par es siempre **`(x, y_true)`**. Pasar de uno a otro (dividir entre 255,
one-hot con `n` clases) lo hace quien consume la secuencia, no el adaptador: un adaptador que
supiera cuántas clases hay estaría mezclando formato con modelo.

### La forma de la entrada es propiedad del dato

IDX la trae en la cabecera. Un CSV no: una fila es una lista plana de números y `(28, 28)` o
`(784,)` serían igual de válidos. Por eso el contrato expone `x_shape` sin decir de dónde sale, y
cada adaptador la obtiene a su manera.

### Cómo se comprueba que el port no depende de la infraestructura

Decirlo no basta; se comprueba de tres maneras:

- **Una batería de contrato** ([tests/contract](../../tests/contract/test_raw_sample_sequence_contract.py)):
  los mismos tests para IDX, CSV y un adaptador en memoria sin ficheros. Un adaptador nuevo se
  registra en una tabla y pasa exactamente los mismos casos.
- **Fronteras de importación** ([tests/architecture](../../tests/architecture/test_import_boundaries.py)):
  `domain` y `ports` solo pueden importar lo que una **lista de permitidos** autoriza. Se
  eligió lista de permitidos y no de prohibidos porque esta última no detecta lo que no se nos
  ocurrió prohibir.
- **Los ficheros reales**: los tests sintéticos usan un helper y un adaptador escritos por la
  misma cabeza, así que un malentendido común del formato los pasaría. Los tests contra los
  ficheros reales comparan con literales verificados por fuera (sección 11).

### El cierre de recursos es del adaptador, no del contrato

Un adaptador que abre ficheros es un context manager: si algo falla al abrir no queda ningún
fichero abierto (`ExitStack`), y `with` cierra también si el cuerpo lanza. El contrato solo
declara `len`, `x_shape` y `x[i]`: un adaptador en memoria no tiene nada que cerrar.

## 5. El formato IDX en cinco minutos

Los ficheros originales de MNIST son IDX. Merece la pena mirarlos con un volcado
hexadecimal antes de escribir una sola línea:

```text
$ xxd -l 16 train-labels-idx1-ubyte
00000000: 0000 0801 0000 ea60 0500 0401 0902 0103
          └─magic──┘ └──N───┘ └── primeras etiquetas: 5 0 4 1 9 2 1 3

$ xxd -l 16 train-images-idx3-ubyte
00000000: 0000 0803 0000 ea60 0000 001c 0000 001c
          └─magic──┘ └──N───┘ └─ 28 ──┘ └─ 28 ──┘
```

| Bytes | Significado |
|---|---|
| `00 00` | Siempre cero. |
| `08` | Tipo de dato: `0x08` = *unsigned byte*, el único que usa MNIST. |
| `01` / `03` | Número de dimensiones (`ndims`): 1 para etiquetas, 3 para imágenes. |
| `0000 ea60` | Primera dimensión, `N`, en **big-endian**: `0xea60` = 60000. |
| `0000 001c` | Siguientes dimensiones: `0x1c` = 28. |

Tras la cabecera vienen los datos, con **el último índice variando más rápido** (para una
imagen de `(28, 28)`, fila a fila). Todos los registros miden lo mismo, así que la posición del
ejemplo `i` se calcula:

```text
posición(i) = tamaño_de_cabecera + i × tamaño_de_registro
tamaño_de_cabecera = 4 + 4 × ndims
tamaño_de_registro = producto(dims[1:]) × 1 byte
```

Y el tamaño del fichero se puede **predecir** y comprobar:

| Fichero | Cálculo | Bytes |
|---|---|---|
| imágenes de entrenamiento | `16 + 60000 × 784` | 47.040.016 |
| etiquetas de entrenamiento | `8 + 60000` | 60.008 |
| imágenes de prueba | `16 + 10000 × 784` | 7.840.016 |
| etiquetas de prueba | `8 + 10000` | 10.008 |

Dos detalles con consecuencias reales:

- **La endianness.** El 256 se escribe `00 00 01 00`; leído como little-endian sale 65536. Por eso
  los tests usan `N = 256`: un error de endianness se ve al instante.
- **No fiarse de una cabecera que aún no se ha validado.** El adaptador comprueba magic, tipo y
  `ndims`, y contrasta el tamaño real del fichero con `cabecera + N × registro` *antes* de usar
  las dimensiones para posicionarse.

Un `.gz` no se lee: falla por magic incorrecto (`0x1f8b0808`). El acceso por posición exige un
fichero descomprimido, porque en un flujo gzip no se puede saltar al byte `cabecera + i × tamaño`
sin descomprimir todo lo anterior.

## 6. CSV: lo que IDX daba gratis

> Lo que sigue sobre las variantes del CSV de MNIST es de memoria y **no se ha verificado**:
> ningún CSV real de MNIST está en el repositorio.

El MNIST en CSV existe porque el IDX es incómodo: es binario, propio y viene comprimido.
Cualquier herramienta abre un CSV. La disposición habitual es una fila por imagen, con la
etiqueta en la primera columna y 784 píxeles como enteros de 0 a 255. Hay al menos dos variantes
conocidas que **difieren en la cabecera**: la conversión de Joseph Redmon no la tiene, y la de la
competición *Digit Recognizer* de Kaggle sí. Por eso "el CSV habitual" no estaba definido, y se
eligió sin cabecera y con la etiqueta primero.

Lo que IDX resolvía y el CSV no:

| Aspecto | IDX | CSV |
|---|---|---|
| Forma de la entrada | en la cabecera | no está: se pasa al constructor |
| Número de ejemplos | en la cabecera | hay que contar las líneas |
| Tamaño de registro | fijo | variable (`0` ocupa 1 carácter, `255` ocupa 3) |
| Rango 0–255 | lo garantiza el tipo | hay que validarlo al leer |

**Consecuencias de diseño:**

- **La forma es un parámetro** de `CsvRawSampleSequence`, y se comprueba que
  `1 + producto(forma)` coincide con el número de columnas de cada fila. Sin esa comprobación
  la forma sería decorativa: un `(28, 27)` pasaría sin quejas.
- **No hay fórmula `cabecera + i × tamaño`**, así que al abrir se recorre el fichero una vez y se
  guarda **solo dónde empieza cada línea** (no el contenido). Costó una decisión explícita:
  cargar todo el fichero en memoria habría sido más sencillo, pero gasta tanta memoria como el
  fichero.
- **Se rechazó un fichero de metadatos aparte** (forma y número de filas en un JSON): habría
  inventado un formato que nadie produce, y el CSV real no se podría leer sin escribirle a mano
  su JSON. Si algún día se quiere, sería otro adaptador.
- **El contenido de las celdas se valida al leer `x[i]`, no al abrir.** Un píxel corrupto se
  descubre cuando alguien lee esa fila. Las líneas en blanco al final se toleran; en medio son un
  error; un fichero sin ejemplos se rechaza.

**Limitaciones que quedan documentadas en el propio adaptador:** `int()` acepta `1_0` (que vale
10), ` 5` y `+5`, y rechaza `255.0`.

## 7. Cómo trabajamos

Esta capa se construyó en sesiones de trabajo en pareja con un asistente de IA (Claude). Las
reglas que se fijaron al principio fueron parte de lo aprendido:

- **Test primero, en rojo; después la implementación mínima que lo pone en verde.** Y el rojo
  tiene que ser **por la razón correcta**: un `ModuleNotFoundError` vale para empezar; un test
  que falla porque el propio test está mal, no.
- **Un paso por vez.** Al terminar cada uno, un resumen corto y esperar confirmación.
- **Puntos de parada** (*PARADA*) donde la decisión es del autor y no se sigue sin su respuesta.
- **Crítica sin concesiones.** Marcar como **hipótesis** todo lo que no se haya verificado.
- **Nombres honestos y solo librería estándar** (`struct`, `contextlib`, `math`, `ast`...).

**Mutaciones a mano.** Un test que nunca ha fallado no demuestra nada. Después de poner un test
en verde, se rompe el código a propósito (cambiar `>` por `<`, quitar una comprobación, desplazar
un índice) y se comprueba que el test cae. Luego se restaura y se compara con `diff`. Casi cada
sección de la bitácora tiene una mutación detrás.

**Los tests que nacen en verde** son frecuentes y hay que decirlo. Si una validación ya la cubre
otra, el test nuevo pasa sin pasar por rojo: no empujó ninguna línea de código. Se dejan cuando
documentan el contrato, y se anota que nacieron en verde. Un ejemplo: "etiquetas con forma
`(N, 1)`" no es una validación independiente, porque con `ndims` fijo en 1 ya la cubre la de
`ndims`.

## 8. La bitácora, paso a paso

Cada fila es un paso o un grupo de commits. El número de paso es el del plan de la sesión.

| Paso | Commits | Qué se hizo | Lo que enseñó |
|---|---|---|---|
| Renombrado | `ee7639e` | `Sample` pasa a `NetSample`. | Un nombre debe decir en qué estado está el dato: ya transformado. |
| 3. Tipos y port | `7a48e8d` | `RawSample`, `NetSample` y el port, solo firmas y docstrings. | El contrato vive en los docstrings: orden del par, qué es "raw", la convención de MNIST frente a la de IDX. |
| Vector | `776d28f` `a7714bb` `b7aafac` | Se hace hashable, después inmutable, y se arregla un bug del generador. | Pedir samples `frozen` destapó tres deudas en `Vector` (sección 9). |
| 4. Helper de tests | `630843f` | `write_idx`, genérico por dimensiones. | Es una segunda implementación del formato, escrita por quien escribe el adaptador. Su límite quedó escrito en el docstring. |
| 5.1 y 5.2 | `32fa11f` `1a780fb` | `len` y `x_shape` salen de la cabecera. | Elegir `N = 256` y `(2, 3, 40)` para que un error de endianness o de orden no pase por casualidad. |
| 5.3 | `b1e9615` | Validaciones al abrir: magic, tipo, `ndims`, tamaños, N distinto, `.gz`. | Un fichero más corto que su cabecera daba `struct.error`, no un error útil. |
| 5.5 (antes que 5.4) | `aa8b5dd` | `x[i]` con `seek` + `read`. | Se cambió el orden: el 5.4 habría nacido en verde mientras no se retuvieran los handles. |
| 5.4 | `618592a` | Cerrar todo lo abierto si falla al abrir. | `ExitStack`, y un espía de `open` que además afirma cuántas aperturas ve. |
| 5.6 | `e435728` | `with`: `__enter__` devuelve `self`. | El cierre es del adaptador que posee ficheros, no del port. |
| 6. Batería de contrato | `c4307bb` | Los tests del port pasan a una batería con un adaptador en memoria. | El doble ingenuo cayó justo con los índices negativos. |
| 6b. Fronteras | `3a4207e` | Test con `ast` sobre `domain` y `ports`. | Lista de permitidos, y un test que sabe que puede fallar. |
| 6c. Adaptador CSV | `f89f4ae` `6746bd5` | `CsvRawSampleSequence` y sus tests. | Lo que IDX daba gratis (sección 6). |
| 7. Ficheros reales | `e4d15ff` `8ef3f0f` | Configuración de rutas y tests contra los ficheros reales. | Los literales salen de fuera: `stat` y un `xxd` hecho a mano. |
| Retirar el camino viejo | `3c61e6a` | Se eliminan `SampleSource` e `IdxMnistSource`. | Se documenta adónde fue cada comportamiento. |
| Pydantic | `4733641` `30bba14` | `Settings` con la librería estándar y dependencias fuera. | Un aviso de obsolescencia como excusa para preguntarse si la dependencia hacía falta. |
| 8. mmap | *(cancelado)* | Se descartó un adaptador mmap y su medición. | mmap y `seek` + `read` son la misma infraestructura: no habría demostrado nada sobre la independencia del port. |

### El camino antiguo, retirado

Antes existía `SampleSource`, con `load() -> Iterator[NetSample]`, y su adaptador `IdxMnistSource`,
que recibía los ficheros ya leídos como `bytes`. Se retiró porque combinaba tres cosas que el
diseño nuevo separa:

- **Iteración de un solo uso.** Sin `len`, sin acceso por índice y sin barajar, quien necesitara
  algo de eso debía materializar todo con `list(source.load())`: 60.000 `Vector` de 784 flotantes
  de Python en memoria (coste no medido).
- **El fichero entero en memoria** como `bytes`.
- **Decisiones de modelo dentro del adaptador:** dividía entre 255 y hacía el one-hot con 10
  clases fijas.

| Antes | Ahora |
|---|---|
| `parse_idx_header` y las comprobaciones de tamaño y de N | validaciones al abrir de `IdxRawSampleSequence` |
| Normalizar entre 0 y 1, y one-hot con 10 clases | pendiente del consumidor de la secuencia, que produce `NetSample` |
| Recorrer con `load()` una vez por época | recorrer índices sobre la secuencia |
| Fuente a partir de `bytes` en memoria | el adaptador en memoria de los tests, solo para pruebas |

Los tests del contrato de `load()` (iterador nuevo en cada llamada, iterador que se agota,
`islice`) **no tienen equivalente a propósito**: describen una semántica de iteración que el
diseño elimina. `NetSample` se conserva porque lo usan los playgrounds.

El código retirado sigue en el historial:
`git show 8ef3f0f:src/mnist/adapters/idx_mnist_source.py`.

## 9. Trampas y sorpresas

Lo más útil de una bitácora es lo que no salió como se esperaba.

| # | Qué pasó | Qué enseña |
|---|---|---|
| 1 | **Los índices negativos.** En el adaptador IDX ingenuo, `secuencia[-1]` **leía bytes de la cabecera como si fueran una imagen**, sin fallar. `n` y `n+1` devolvían bytes vacíos, y `-n` daba `OSError`. Después, el doble en memoria y el CSV ingenuos cayeron con `-1` y `-n`, porque una lista los acepta. | Un contrato que se desvía de Python hay que ponerlo en una batería común: cada implementación lo olvida sola. |
| 2 | **`Vector` no era hashable** y por eso `Matrix.__hash__` llevaba tiempo roto, sin tests que lo cubrieran. Hacer `frozen` a los samples lo destapó. | Una propiedad de un tipo se descubre cuando otro tipo la exige. |
| 3 | **Cerrar `__setattr__` rompía `deepcopy` y `pickle`**, que restauran el estado con `setattr`. Se planteó como hipótesis, se confirmó ejecutando y se arregló con `__reduce__`. | Contrastar una sospecha antes de arreglarla. |
| 4 | **`Vector(x for x in ...)` salía vacío**: el bucle de validación consumía el generador. | Un iterable de un solo uso no se recorre dos veces. |
| 5 | **Un falso rojo tras una mutación.** Se restauró el código y el test seguía fallando: un `.pyc` obsoleto (mismo tamaño y mismo segundo de modificación) hacía que Python no lo recompilase. | Al mutar código, ejecutar sin bytecode y borrar los `.pyc` al restaurar. |
| 6 | **Una mutación que no verificó nada.** Una violación de import en `models.py` creó un import circular que rompió `conftest.py` antes de correr un solo test, y la salida vacía no era un éxito. | Una salida vacía no es un resultado: hay que ver el resumen. |
| 7 | **`int()` acepta `1_0`, ` 5` y `+5`** en las celdas del CSV. | Hacer "validación gratis" con `int()` es menos estricto de lo que parece. |
| 8 | **Un test del CSV que no muerde.** El de lectura con fin de línea Windows pasa aunque el adaptador no quite el `\r`, porque `int()` ya lo tolera. | Un test puede ser correcto y aun así no proteger la línea que crees. |
| 9 | **`uv remove` desinstaló los extras de desarrollo** (`pytest`, `mypy`, `ruff`, `black`) al resincronizar el entorno solo con las dependencias por defecto. Se vio porque la salida listaba paquetes que no eran de pydantic. | Leer la salida completa de una herramienta que modifica el entorno. Se restauró con `uv sync --extra dev`. |
| 10 | **Una estimación equivocada.** Se predijeron "decenas de segundos" para una comprobación estadística; medida, tardó 0,9 s. | Marcar lo no medido como hipótesis, y medir antes de decidir. |
| 11 | **Un error de descripción.** Se llamó "el CSV habitual" al que no lleva cabecera, y hay una variante muy usada que sí la lleva. | "Lo habitual" no es una especificación. |
| 12 | **mmap deja un descriptor propio.** Un `mmap` mantiene su propio descriptor de fichero, y un `ResourceWarning` no lo detecta. Mirado para el adaptador que se canceló. | Comprobar la liberación de recursos es más sutil que ver un `.closed`. |

## 10. Decisiones y alternativas descartadas

| Decisión | Alternativas | Por qué |
|---|---|---|
| Campos `x` e `y_true` en `RawSample` y `NetSample` | `entrada` y `objetivo` | Un solo par de nombres, coherente con el código existente. Renombrar `NetSample` tocaba unas 30 líneas. |
| Samples `frozen` | Mutables | Un ejemplo leído no debe cambiar bajo quien lo tiene. |
| `RawSample` y `NetSample` juntos en `domain/samples.py` | Dejarlos en `models.py` | Elección del autor: son dos estados del mismo dato. |
| Cerrar con `ExitStack` | `try/except` en `__init__`, o una fábrica | El fallo puede ocurrir tras abrir **los dos** ficheros; `ExitStack` cierra lo que esté abierto. |
| Observar el cierre con un espía de `open` (`monkeypatch`) | Inyectar la función de apertura; contar descriptores en `/dev/fd` | No toca producción. Límite: solo ve llamadas al nombre `open`, y por eso cada test afirma cuántas aperturas espera. |
| Fronteras con **lista de permitidos** | Lista de prohibidos | Una lista de prohibidos da falsa seguridad. |
| Adaptador en memoria en `tests/` | En `src/` | En producción no tendría consumidor. |
| Forma del CSV como parámetro | Un fichero de metadatos aparte | Inventaría un formato que nadie produce. |
| Contenido del CSV validado **al leer**, con `ValueError` | Validarlo todo al abrir | Al abrir ya hay una pasada obligatoria; parsear cada entero la haría mucho más cara. |
| Validar solo cabecera y tamaño al abrir el IDX | Nada, o todo | Es coste constante y evita los errores humanos más probables. |
| `.gz` rechazado | Soportarlo | Exige otro modelo de acceso: no se puede saltar a una posición sin descomprimir. |
| mmap cancelado | Implementarlo y medir | Misma infraestructura que `seek` + `read`. |
| Configuración con dataclass y `os.environ` | Migrar a `model_config` de pydantic | Nadie leía tres de los campos y `.env.dev` no existía. |
| Orden 5.5 antes de 5.4 | El orden del plan | El 5.4 habría nacido en verde sin handles retenidos. |

## 11. Qué está verificado y qué es hipótesis

### Verificado por tests (en la suite)

- **Tipos:** campos y orden, inmutabilidad, hash de `NetSample`. `Vector`: hashable, inmutable,
  con `deepcopy` y `pickle`.
- **Contrato del port**, con IDX, CSV y el adaptador en memoria: `len`, `x_shape`, `x[i]` con
  imagen y etiqueta alineadas, bordes 0 y `n-1`, `IndexError` para `-1`, `-n`, `n` y `n+1`, y
  lecturas repetidas sin estado.
- **Adaptador IDX:** todas las validaciones al abrir, big-endian, cierre de recursos ante fallo y
  con `with`.
- **Adaptador CSV:** ancho frente a la forma, filas desiguales, líneas en blanco, fichero sin
  ejemplos, celdas inválidas al leer, CRLF y cierre de recursos.
- **Fronteras:** `domain` y `ports` solo importan lo permitido.
- **Ficheros reales:** tamaños, `len` de 60.000 y 10.000, forma `(28, 28)`, las 8 primeras
  etiquetas de entrenamiento y de prueba **frente a un volcado `xxd` hecho a mano**, y la primera
  y la última muestra con 784 bytes.

### Verificado por ejecución, fuera de la suite

- **Que la imagen `i` es el dígito de la etiqueta `i`.** Leyendo los ficheros con cortes directos
  de bytes (sin el adaptador): las tres primeras imágenes, dibujadas en ASCII, eran un 5, un 0 y
  un 4, y un clasificador por **centroide** (sección 15) acertó el **77,4 %** con las etiquetas
  alineadas frente al **13,8 %** con las etiquetas desplazadas una posición (el azar ronda el
  10 %). Tardó 0,9 s.
- Los comportamientos de `int()` con celdas raras, y el efecto de `uv remove` sobre el entorno.

### Sigue siendo hipótesis

- **El modelo mental compartido.** El helper de IDX y el adaptador salen de la misma cabeza. Los
  ficheros reales lo cubren para cabecera, tamaño y etiquetas.
- **La alineación a través del adaptador.** La prueba del centroide leyó los ficheros, no el
  adaptador, y no está en la suite.
- **Un CSV real de MNIST:** nunca se ha leído uno. Las variantes de Redmon y de Kaggle son de
  memoria.
- **Sin medir:** el tiempo de la pasada inicial del CSV, la memoria de su índice y el tamaño de
  un CSV real.
- **Ficheros que cambian tras abrirlos:** ninguno de los adaptadores lo detecta y no hay test.
- **Usar la secuencia tras cerrarla:** se observó `ValueError`, pero ningún test lo fija.
- **Las fronteras de importación** no ven `importlib` ni `__import__`, solo cubren `domain` y
  `ports`, y la lista de permitidos es criterio del autor.
- **Las mutaciones** se hicieron a mano; no forman parte de la suite.
- **Solo macOS.**

## 12. Pendiente

**Una herramienta aparte que valide los ficheros de datos antes de entrenar.** No se valida el
contenido cada vez que se abre un fichero: eso sería una locura. Lo que hoy se comprueba al abrir
es estructural (en IDX, unos pocos bytes y un `fstat`; en CSV, una pasada que ya es necesaria
para el índice). La validación de verdad será una herramienta que se ejecute al preparar el
laboratorio, donde decenas de segundos son asumibles.

Ideas para cuando se construya, que son recomendaciones y no decisiones:

- **Que consuma el port**: así vale para IDX, para los CSV que se encuentren y para los datos
  que genere el autor. Además, los datos generados **tendrán que cumplir** lo que la herramienta
  exija: la herramienta es la puerta de aceptación.
- **Dos niveles:** una pasada estructural completa (parsear cada celda del CSV) y la comprobación
  estadística por centroide.
- **Números y no solo pasa o falla:** la precisión con las etiquetas reales **frente a un
  control** con las etiquetas barajadas. Los umbrales fijos son arbitrarios; lo que informa es
  la diferencia.
- **Lo que no detectaría:** un desplazamiento en unas pocas posiciones sueltas, un ruido de
  etiquetas bajo, imágenes duplicadas o un solo píxel corrupto. Es un estadístico agregado.

**Fase de eficiencia.** El CSV recorre el fichero entero en cada apertura: hay que medirlo, y se
podría guardar el índice en disco. También, cuánto cuesta cada acceso por posición.

**El consumidor de la secuencia** (el muestreador): pasar de `RawSample` a `NetSample`
(normalizar, one-hot) y barajar índices. Hoy ningún código de `src` produce un `NetSample`.

**Probar con un CSV real de MNIST**, cuando se encuentre uno.

## 13. Cómo convertirlo en una ruta didáctica

Cada paso de la bitácora es una lección posible. Una propuesta de orden, con la trampa real que
enseña cada una:

| Lección | Se aprende | Artefacto | La trampa |
|---|---|---|---|
| 1. Mirar los bytes | Leer un IDX con `xxd`: magic, `ndims`, big-endian | Los dos volcados de la sección 5 | Predecir el tamaño del fichero antes de comprobarlo |
| 2. Contrato antes que código | Tipos, un port y sus docstrings | `RawSample`, `NetSample`, `RawSampleSequence` | Un contrato que se desvía de Python hay que decirlo |
| 3. Test primero, en rojo | Ciclo rojo → verde → mutación | Las validaciones al abrir | Los tests que nacen en verde; el `.pyc` obsoleto |
| 4. Los límites de un helper | Una segunda implementación no es un oráculo | `write_idx` | El modelo mental compartido |
| 5. Recursos | `with`, `ExitStack`, observar un cierre | El espía de `open` | Cerrar lo abierto cuando falla el segundo fichero |
| 6. Un contrato, varias implementaciones | Batería parametrizada | El adaptador en memoria | Las listas aceptan índices negativos |
| 7. Otro formato | Qué daba gratis el formato anterior | El adaptador CSV | El índice de líneas; `int()` demasiado permisivo |
| 8. Arquitectura comprobable | Fronteras con `ast` | `import_rules.py` | Lista de permitidos frente a prohibidos |
| 9. Contrastar con el mundo real | Literales que salen de fuera | Los tests con ficheros reales; el centroide | Que un dato sea distinto de lo que crees |
| 10. Retirar código | Documentar el porqué | La sección "El camino antiguo, retirado" | Los tests que no se portan a propósito |

## 14. Cómo ejecutarlo

```text
uv sync --extra dev          # instala también pytest, mypy, ruff y black
uv run pytest                # 167 tests
uv run mypy                  # comprobación de tipos
```

> **Cuidado.** `uv add` y `uv remove` resincronizan el entorno solo con las dependencias por
> defecto y **desinstalan los extras de desarrollo**. Si pasa, `uv sync --extra dev` los repone.

Los tests contra los ficheros reales leen las rutas de `config/settings.py`, que por defecto
apuntan a `playgrounds/original_mnist/datasets/` (los ficheros descomprimidos los ignora git;
salen de los `.gz` versionados). Cada ruta se puede sobrescribir con una variable de entorno:
`TRAIN_IMAGES_PATH`, `TRAIN_LABELS_PATH`, `T10K_IMAGES_PATH` y `T10K_LABELS_PATH`. Si un fichero
falta, el test se salta y el mensaje dice qué ruta buscó y qué variable la cambia.

## 15. Glosario

| Término | Significado |
|---|---|
| **Port** | Un contrato que el dominio define para hablar con el exterior, sin saber cómo se cumple. |
| **Adaptador** | Una implementación de un port sobre una infraestructura concreta (un formato de fichero, la memoria). |
| **Raw** | Sin transformar para el modelo: los bytes tal como están guardados. |
| **Magic** | Los cuatro primeros bytes de un IDX: dos ceros, el tipo de dato y el número de dimensiones. |
| **Big-endian** | Orden de los bytes de un entero con el más significativo primero: `00 00 01 00` es 256. |
| **Registro** | Los bytes de un ejemplo. En IDX miden todos lo mismo, y de ahí sale la posición de cada uno. |
| **Índice de desplazamientos** | La lista de en qué byte empieza cada línea de un CSV, para saltar a ella sin releer el fichero. |
| **Centroide** | El punto medio de un grupo. Aquí, para cada dígito, la imagen media de todas las imágenes con esa etiqueta. Clasificar es elegir el dígito cuya imagen media queda más cerca. Si las etiquetas están alineadas con las imágenes acierta en torno al 77 %; si no, se queda en el azar. |
| **Nacer en verde** | Un test nuevo que pasa sin haber pasado por rojo, porque otra pieza de código ya cubría lo que comprueba. |
| **Mutación** | Romper el código a propósito para comprobar que un test lo detecta. |
