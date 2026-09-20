"""Que puede importar cada capa, y un comprobador basado en ast.

Es una lista de PERMITIDOS: un import nuevo en domain o ports hace fallar el test y obliga
a decidir a conciencia si la capa puede depender de eso. Una lista de prohibidos daria
falsa seguridad, porque no detectaria lo que no se nos ocurrio prohibir.

Limite: solo ve sentencias import. No ve importaciones dinamicas (importlib, __import__).
"""
import ast
from collections.abc import Iterator

# Prefijos permitidos: `typing` deja pasar `typing.Protocol`, `mnist.domain` deja pasar
# `mnist.domain.samples`, pero `mnist.adapters` no queda cubierto por ninguno.
RULES: dict[str, frozenset[str]] = {
    "domain": frozenset(
        {"abc", "collections.abc", "dataclasses", "functools", "math", "random", "typing",
         "mnist.domain"}
    ),
    "ports": frozenset({"typing", "mnist.domain", "mnist.ports"}),
}


def _imported_names(source: str, module_name: str, is_package: bool) -> Iterator[str]:
    """Cada nombre importado, con los imports relativos ya resueltos a absolutos.

    `from a import b` produce `a.b`: asi `from mnist import adapters` se ve como
    mnist.adapters aunque el modulo importado sea solo `mnist`.
    """
    package = module_name if is_package else module_name.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            for alias in node.names:
                yield f"{module}.{alias.name}"


def violations(source: str, module_name: str, layer: str, is_package: bool = False) -> list[str]:
    """Nombres importados por `source` que la capa no tiene permitidos."""
    allowed = RULES[layer]
    return sorted(
        {
            name
            for name in _imported_names(source, module_name, is_package)
            if not any(name == prefix or name.startswith(prefix + ".") for prefix in allowed)
        }
    )
