import numpy.linalg
import scipy.linalg


def PLUsolve(A):
    """
    Return solver suitable for A*x=b for map(), where the approximate solution is computed in
    floating point arithmetic by scipy's wrapper for LAPACK DGETRF.
    Returned solver maps right hand side b to soln x for fixed coefficient matrix A.
    :param A:
    :return:
    """

    # Compute L U factorization of A via wrapper of LAPACK DGETRF
    LU, P = scipy.linalg.lu_factor(A)

    def rhs_solver(b):
        return scipy.linalg.lu_solve((LU, P), b)

    return rhs_solver
