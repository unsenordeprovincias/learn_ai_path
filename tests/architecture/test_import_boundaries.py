"""La arquitectura hexagonal, comprobada sobre los imports: domain y ports no dependen
de infraestructura. Es una comprobacion de la DIRECCION de las dependencias, no de la
semantica del contrato (eso lo hace la bateria de tests/contract)."""
from pathlib import Path

import pytest

from tests.import_rules import RULES, violations

SRC = Path(__file__).resolve().parents[2] / "src"


def test_un_import_de_infraestructura_se_detecta():
    fuente = "import struct\nfrom typing import Protocol\n"

    assert violations(fuente, "mnist.ports.x", "ports") == ["struct"]


def test_un_import_relativo_hacia_los_adaptadores_se_detecta():
    # `from ..adapters import ...` desde mnist/ports/x.py es mnist.adapters
    fuente = "from ..adapters import idx_raw_sample_sequence\n"

    assert violations(fuente, "mnist.ports.x", "ports") == ["mnist.adapters.idx_raw_sample_sequence"]


def test_lo_permitido_no_se_marca():
    fuente = (
        "from collections import abc\n"
        "from mnist.domain.samples import RawSample\n"
        "from .models import Vector\n"
    )

    assert violations(fuente, "mnist.domain.x", "domain") == []


@pytest.mark.parametrize("capa", RULES)
def test_los_modulos_de_la_capa_solo_importan_lo_permitido(capa):
    ficheros = sorted((SRC / "mnist" / capa).glob("*.py"))
    assert ficheros, f"no se encontro ningun fichero en mnist/{capa}: el test estaria en vacio"

    encontrados = {}
    for fichero in ficheros:
        nombre = ".".join(fichero.relative_to(SRC).with_suffix("").parts)
        if fichero.name == "__init__.py":
            nombre = nombre.removesuffix(".__init__")
        prohibidos = violations(fichero.read_text(), nombre, capa, is_package=fichero.name == "__init__.py")
        if prohibidos:
            encontrados[fichero.name] = prohibidos

    assert encontrados == {}
