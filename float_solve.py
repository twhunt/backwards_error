import numpy.linalg
import scipy.linalg


def LUsolve(A, rhss):

    ###################################################################################
    # Compute L U factorization of A via wrapper of LAPACK DGETRF
    LU, P = scipy.linalg.lu_factor(A)

    return [scipy.linalg.lu_solve((LU, P), rhs) for rhs in rhss]

