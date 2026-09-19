# ports/sample_source.py
from typing import Protocol
from mnist.domain.models import Sample


class SampleSource(Protocol):
    def load(self) -> list[Sample]:
        ...