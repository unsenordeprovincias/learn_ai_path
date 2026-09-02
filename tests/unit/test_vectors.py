from mnist.domain.models import Vector
import pytest
import random

def test_create_vector():
    v = Vector([1, 2, 3])
    assert isinstance(v, Vector)
    assert len(v) == 3
    assert v[0] == 1
    assert isinstance(v.values, tuple)

    #more pythonic
    v1 = Vector[2, 4, 6]
    assert isinstance(v1, Vector)
    assert len(v1) == 3
    assert v1[0] == 2
    assert isinstance(v1.values, tuple)

    assert v1 == Vector([2, 4, 6])

def test_create_vector_from_values():
    v = Vector.from_values(1, 3, 2)
    assert isinstance(v, Vector)
    assert len(v) == 3
    assert v[0] == 1

def test_items_of_vector_must_be_floats():
    with pytest.raises(TypeError) as exception_info:
        Vector([1, 'a', 3])
    assert "must be float" in str(exception_info.value)
    assert exception_info.type == TypeError

    with pytest.raises(TypeError) as exception_info:
        Vector.from_values(1, 'a', 3)
    assert "must be float" in str(exception_info.value)
    assert exception_info.type == TypeError


def test_vectors_identity():
    v1 = Vector([1, 2, 3])
    v2 = Vector.from_values(1, 2, 3)

    assert v1 == v2

def test_adding_vectors():
    v1 = Vector([1, 2, 3])
    v2 = Vector.from_values(1, 2, 3)

    v3 = v1 + v2
    assert isinstance(v3, Vector)
    assert v3 == Vector([2, 4, 6])

def test_substracting_vectors():
    v1 = Vector([1, 2, 3])
    v2 = Vector.from_values(2, 4, 6)

    v3 = v1 - v2
    assert isinstance(v3, Vector)
    assert v3 == Vector([-1, -2, -3])


def test_size_error_adding_vectors():
    with pytest.raises(ValueError) as exc_info:
        Vector([1, 2]) + Vector([1, 2, 3])
    assert "with shapes (2,) (3,)" in str(exc_info.value)

def test_type_errors_operations_with_vectors():
    v1 = Vector([1, 2, 3])
    v2 = v1 * 2

    with pytest.raises(TypeError):
        v1 + 3

    with pytest.raises(TypeError):
        v1 @ 3

def test_vectors_products():
    v1 = Vector([1, 2, 3])

    v2 = v1 * 2
    assert v2 == Vector([2, 4, 6])

    v2 = 2 * v1
    assert v2 == Vector([2, 4, 6])

    v3 = v1 @ v2
    assert v3 == 28

def test_size_error_matmul_vectors():
    with pytest.raises(ValueError) as exc_info:
        Vector([1, 2]) @ Vector([1, 2, 3])
    assert "with shapes (2,) (3,)" in str(exc_info.value)


def test_str_vector():
    v1 = Vector.from_values(*range(3))
    assert str(v1) == "Vector (0, 1, 2)"


def test_initialize_vector():
    random.seed(7)
    n = 5
    values = [random.uniform(-1, 1) for i in range (n)]

    random.seed(7)
    v = Vector.initialize(n)
    for value, component in zip(values, v.values):
        assert component == value

def test_one_hot():
    assert Vector.one_hot(0, 3) == Vector[1, 0, 0]
    assert Vector.one_hot(1, 3) == Vector[0, 1, 0]
    assert Vector.one_hot(2, 3) == Vector[0, 0, 1]