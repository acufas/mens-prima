from math import fsum


def dot(a: list[float], b: list[float]) -> float:
    """Return the dot product of vectors a and b.

    Raises ValueError if a and b are not the same length.
    """
    if len(a) != len(b):
        raise ValueError("Vectors are not of equal size!")
    n = len(a)
    return fsum(a[i] * b[i] for i in range(n))


def _shape(matrix: list[list[float]]) -> tuple[int, int]:
    """Return the (rows, columns) shape of a matrix.

    Raises ValueError if the matrix is ragged.
    """
    if not matrix:
        return (0, 0)
    m = len(matrix)
    n = len(matrix[0])
    for i in range(1, m):
        if len(matrix[i]) != n:
            raise ValueError(f"Ragged matrix: row {i} has {len(matrix[i])} columns, expected {n}")
    return (m, n)


def matvec(
    matrix: list[list[float]],
    vector: list[float]
) -> list[float]:
    """Return the product of an m x n matrix and a length-n vector.

    Each entry of the resulting length-m vector is the dot product of
    one row of the matrix with the vector.

    An empty matrix returns an empty vector for any vector. A matrix
    with zero columns and an empty vector returns one 0.0 per row.

    Raises ValueError if the matrix is ragged, or if the number of
    matrix columns does not equal the length of the vector.
    """
    m, n = _shape(matrix)
    if m == 0:
        return []
    if n != len(vector):
        raise ValueError(f"Unequal inner dimensions: matrix has {n} columns but vector has length {len(vector)}")

    return [dot(row, vector) for row in matrix]
