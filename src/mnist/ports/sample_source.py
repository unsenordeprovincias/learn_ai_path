# ports/sample_source.py
from typing import Iterator, Protocol
from mnist.domain.models import Sample


class SampleSource(Protocol):
    def load(self) -> Iterator[Sample]:
        """Devuelve un iterador nuevo y de un solo uso en cada llamada.

        Para recorrer los datos otra vez (otra época) hay que volver a llamar
        a load(). Quien necesite len(), barajar o acceso por índice decide
        materializarlo con list(source.load()).
        """
        ...
