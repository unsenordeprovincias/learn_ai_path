"""Adaptador en memoria del port RawSampleSequence, un doble para los tests.

No lee ficheros, no tiene cabecera ni magic ni ndims: existe para comprobar que la
bateria de contrato no asume nada de IDX. Vive en tests/ porque en produccion no
tendria ningun consumidor.
"""
from mnist.domain.samples import RawSample


class InMemoryRawSampleSequence:
    def __init__(self, samples: list[RawSample], x_shape: tuple[int, ...]):
        self._samples = samples
        self._x_shape = x_shape

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def x_shape(self) -> tuple[int, ...]:
        return self._x_shape

    def __getitem__(self, i: int) -> RawSample:
        # una lista aceptaria los negativos (x[-1] es el ultimo); el port no
        if not 0 <= i < len(self._samples):
            raise IndexError(f"sample index {i} is out of range for {len(self._samples)} samples")
        return self._samples[i]
