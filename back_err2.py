import numpy.linalg
import scipy.linalg
import sympy


def main():

    num_angles = 128
    angle_factor = 2.0*numpy.pi/num_angles

    rhs_angles = [k*angle_factor for k in range(num_angles)]
    rhss = [numpy.array([[numpy.cos(angle)], [numpy.sin(angle)]]) for angle in rhs_angles]

    U_angle = numpy.pi/3.0
    U_trig = [numpy.cos(U_angle), numpy.sin(U_angle)]
    V_angle = numpy.pi/4.0
    V_trig = [numpy.cos(V_angle), numpy.sin(V_angle)]

    U = rotation_matrix(*U_trig)
    V = rotation_matrix(*V_trig)
    singular_values = [1e2, 1e-6]
    S = numpy.diagflat(singular_values)

    A = numpy.dot(U, S)
    A = numpy.dot(A, V.T)

    LU, P = scipy.linalg.lu_factor(A)

    solns = [scipy.linalg.lu_solve((LU, P), rhs) for rhs in rhss]


    rational_A = exact_A(A)
    U, D, V = exact_svd(rational_A)
    pass

def rotation_matrix(cosine, sine):
    return numpy.array([[cosine, -sine], [sine, cosine]])

def exact_A(A):

    m = len(A)
    n = len(A[0])

    out_A = sympy.zeros(m, n)

    for r in range(m):
        for c in range(n):
            out_A[r, c] = sympy.Rational(A[r, c])

    return out_A

def exact_svd(A):


    U, sing_vals_sqrd_U = singular_vecs(A*A.T)
    V, sing_vals_sqrd_V = singular_vecs(A.T*A)

    sing_vals = [sympy.sqrt(sing_vals_sqrd) for sing_vals_sqrd in sing_vals_sqrd_U]

    D = sympy.zeros(2)
    for k in range(2):
        D[k, k] = sing_vals[k]

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

if __name__ == '__main__':
    main()