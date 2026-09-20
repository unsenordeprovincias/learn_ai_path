"""Espia del `open` de un adaptador, para comprobar que no deja ficheros abiertos."""


def spy_on_open(monkeypatch, module):
    """Sustituye el `open` de `module` por un espia que deja pasar la llamada real.

    Devuelve la lista, que se va llenando, de los ficheros que el modulo abrio. Solo ve
    llamadas al nombre `open(...)` en ese modulo (no Path.open, io.open ni os.open), asi
    que cada test debe afirmar cuantos ficheros espera ver: sin eso, "todos cerrados"
    pasaria en vacio si el codigo dejara de usar ese nombre.
    """
    abiertos = []

    def espia(*args, **kwargs):
        fichero = open(*args, **kwargs)  # aqui `open` es el de builtins: el parche es del modulo
        abiertos.append(fichero)
        return fichero

    monkeypatch.setattr(module, "open", espia, raising=False)
    return abiertos
