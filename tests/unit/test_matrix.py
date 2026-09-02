from mnist.domain.models import Matrix, Vector
import pytest

def test_create_matrix():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert isinstance(m, Matrix)
    assert m.shape == (2, 3)

def test_create_matrix_from_vectors():
    m = Matrix([Vector([1, 2]), Vector([3, 4])])
    assert isinstance(m, Matrix)
    assert m.shape == (2, 2)

def test_rows_must_have_same_length():
    with pytest.raises(ValueError) as exc_info:
        Matrix([[1, 2], [1, 2, 3]])
    assert "same length" in str(exc_info.value)

def test_items_of_matrix_must_be_floats():
    with pytest.raises(TypeError) as exc_info:
        Matrix([[1, 'a'], [3, 4]])
    assert "must be float" in str(exc_info.value)

def test_shape():
    assert Matrix([[1, 2, 3], [4, 5, 6]]).shape == (2, 3)
    assert Matrix([[1], [2], [3]]).shape == (3, 1)
    assert Matrix([]).shape == (0, 0)

def test_str_matrix():
    m = Matrix([[1, 2], [3, 4]])
    assert str(m) == "Matrix ((1, 2), (3, 4))"

def test_matmul_matrix_vector():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 1, 1])

    result = m @ v
    assert isinstance(result, Vector)
    assert result == Vector([6, 15])

def test_size_error_matmul_matrix_vector():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    v = Vector([1, 1])

    with pytest.raises(ValueError) as exc_info:
        m @ v
    assert "with shapes (2, 3) (2,)" in str(exc_info.value)

def test_type_error_matmul_matrix():
    m = Matrix([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        m @ 3

def test_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    t = m.T

    assert isinstance(t, Matrix)
    assert t.shape == (3, 2)
    assert t.rows == (Vector([1, 4]), Vector([2, 5]), Vector([3, 6]))

def test_transpose_twice_returns_equivalent_matrix():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.T.T.rows == m.rows
