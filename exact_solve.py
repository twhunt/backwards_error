import sympy


def solve(dbl_A):
    """
    Return solver for A*x=b suitable for map(), where the exact solution is computed.
    :param dbl_A:
    :return:
    """

    exact_A_inv = exact_A(dbl_A).inv()

    def solve(dbl_rhs):

        return exact_A_inv*exact_A(dbl_rhs)

    return solve


def perturbed_RHS(dbl_A):
    """
    Returns function suitable for map() that returns the exact right hand side bhat,
    given pertrubed solution xhat such that A*xhat=bhat.
    :param dbl_A:
    :return:
    """

    rational_A = exact_A(dbl_A)

    def perturbed_b(perturbed_soln):
        return rational_A*exact_A(perturbed_soln)

    return perturbed_b


def exact_A(A):
    """
    Return a sympy.Matrix instance built from a matrix of floating point numbers.
    The entries in the returned matrix are sympy.Rationals that exactly match the entries of
    the input matrix.
    :param A:
    :return:
    """

    m = len(A)
    n = len(A[0])

    out_A = sympy.zeros(m, n)

    for r in range(m):
        for c in range(n):
            out_A[r, c] = sympy.Rational(A[r, c])

    return out_A


def svd(A, simplify=True):
    """
    Compute the singular value decomposition of the input matrix,
    using a technique that Gene Golub would frown at, but is ok when computed in exact arithmetic.
    :param A:
    :param simplify:
    :return:
    """

    U, sing_vals_sqrd_U = singular_vecs(A*A.T)
    V, sing_vals_sqrd_V = singular_vecs(A.T*A)

    sing_vals = [sympy.sqrt(sing_vals_sqrd) for sing_vals_sqrd in sing_vals_sqrd_U]

    D = sympy.zeros(2)
    for k in range(2):
        D[k, k] = sing_vals[k]

    if simplify:
        return sympy.simplify(U), sympy.simplify(D), sympy.simplify(V)
    else:
        return U, D, V


def singular_vecs(A_prod):

    r1 = A_prod.eigenvects()

    svals_sqrd = [r1[i][0] for i in range(2)]

    perm = [0, 1] if svals_sqrd[0] >= svals_sqrd[1] else [1, 0]

    svals_sqrd = [svals_sqrd[i] for i in perm]

    svecs = [r1[i][2][0] for i in range(2)]
    svecs = [evec/evec.norm() for evec in svecs]

    svecs = [svecs[i] for i in perm]

    U = sympy.Matrix([list(svecs[0]), list(svecs[1])])
    U = U.T

    return U, svals_sqrd


def svd_inv(U, D, V, b):

    x = U.T*b
    for r in range(2):
        x[r, 0] /= D[r, r]

    x = V*x

    return x

