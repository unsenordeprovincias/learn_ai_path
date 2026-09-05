from playgrounds.xor.original import XOR

def test_XOR():
    assert XOR(1, 1) == 0
    assert XOR(1, 0) == 1
    assert XOR(0, 1) == 1
    assert XOR(0, 0) == 0

