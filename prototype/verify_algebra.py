#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
"""CPU-only randomized verification of the separated intersection algebra.

This requires only Python's standard library.  It checks the feature-vector
form against an independent Gaussian-elimination solution of the original
three-equation system.
"""

import math
import random


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(s, a):
    return tuple(s * x for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def separated_solution(A, C, B, O, r):
    """Return (t,u,v) using triangle-only and ray-only precomputations."""
    n = cross(C, B)
    an = dot(A, n)
    pu = cross(B, A)
    pv = cross(A, C)
    m = cross(O, r)

    nt = an - dot(n, O)
    nu = dot(pu, r) - dot(B, m)
    nv = dot(pv, r) + dot(C, m)
    denominator = dot(n, r)
    return (nt / denominator, nu / denominator, nv / denominator)


def direct_solution(A, C, B, O, r):
    """Solve [r,-C,-B] [t,u,v]^T = A-O by Gaussian elimination."""
    q = sub(A, O)
    matrix = [
        [r[0], -C[0], -B[0], q[0]],
        [r[1], -C[1], -B[1], q[1]],
        [r[2], -C[2], -B[2], q[2]],
    ]

    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1.0e-12:
            return None
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]

        for row in range(3):
            if row == column:
                continue
            factor = matrix[row][column]
            matrix[row] = [
                matrix[row][j] - factor * matrix[column][j] for j in range(4)
            ]

    return tuple(matrix[row][3] for row in range(3))


def random_vec(rng):
    return tuple(rng.uniform(-2.0, 2.0) for _ in range(3))


def main():
    rng = random.Random(0xC0DA)
    tested = 0
    maximum_error = 0.0

    while tested < 10_000:
        A, C, B, O, r = (random_vec(rng) for _ in range(5))
        direct = direct_solution(A, C, B, O, r)
        if direct is None:
            continue

        separated = separated_solution(A, C, B, O, r)
        error = max(abs(a - b) for a, b in zip(direct, separated))
        scale_of_answer = max(1.0, *(abs(value) for value in direct))
        if error > 1.0e-9 * scale_of_answer:
            raise AssertionError(
                f"mismatch at case {tested}: direct={direct}, "
                f"separated={separated}, error={error}"
            )
        maximum_error = max(maximum_error, error)
        tested += 1

    print(f"verified {tested:,} nonsingular random systems")
    print(f"maximum absolute coordinate error: {maximum_error:.3e}")


if __name__ == "__main__":
    main()
