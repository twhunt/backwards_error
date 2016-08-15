import sympy

def solve(dbl_A, dbl_rhss):

    rational_A = exact_A(dbl_A)
    rational_A_inv = rational_A.inv()

    exact_solns = []
    for dbl_rhs in dbl_rhss:
        exact_rhs = exact_A(dbl_rhs)
        exact_solns.append(rational_A_inv*exact_rhs)

    return exact_solns

def perturbed_RHS(dbl_A, dbl_solns):

    rational_A = exact_A(dbl_A)
    return [rational_A*exact_A(dbl_soln) for dbl_soln in dbl_solns]

def absolute_differences(exact_solns, prtrbd_solns, exact_rhss, prtrbd_rhss):

    # exact solutions are Sympy matrices of Sympy.Rationals
    # perturbed solutions are composed of double precision floats
    # exact RHS's are composed of double precision floats
    # prtrbd RHS's are Sympy matrices of Sympy.Rationals

    # out_diffs: difference between exact and perturbed solutions, difference between exact and perturbed RHS
    out_diffs = []
    for exact_soln, prtrbd_soln, exact_rhs, prtrbd_rhs in zip(exact_solns, prtrbd_solns, exact_rhss, prtrbd_rhss):
        out_diffs.append((exact_soln - exact_A(prtrbd_soln), exact_A(exact_rhs) - prtrbd_rhs))

    return out_diffs

def relative_errors(abs_diffs, exact_solns, exact_rhss):

    rel_errs = []
    for abs_diff, exact_soln, exact_rhs in zip(abs_diffs, exact_solns, exact_rhss):
        rel_errs.append((abs_diff[0].norm()/exact_soln.norm(), abs_diff[1].norm()/exact_A(exact_rhs).norm()))

    return rel_errs

def exact_A(A):

    m = len(A)
    n = len(A[0])

    out_A = sympy.zeros(m, n)

    for r in range(m):
        for c in range(n):
            out_A[r, c] = sympy.Rational(A[r, c])

    return out_A


def svd(A, simplify=True):


    #TODO: verify that this is the exact SVD, and simplify

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