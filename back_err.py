import sympy
import sympy.polys.polytools
import math
import matplotlib.pyplot
import copy
########################################################################################################################
def unit_circle(num_points):

    dbl_pi = 2*math.pi
    points = []
    for n in xrange(num_points):
        angle = float(n)/num_points*dbl_pi
        points.append([[math.cos(angle)], [math.sin(angle)]])
    return points
########################################################################################################################
def sympy_mat(in_mat):

    num_rows = len(in_mat)
    num_cols = len(in_mat[0])

    out_mat = sympy.Matrix.zeros(num_rows, num_cols)
    for row in xrange(num_rows):
        for col in xrange(num_cols):
            out_mat[row, col] = sympy.Rational(in_mat[row][col])

    return sympy.Matrix(out_mat)
########################################################################################################################
def det_solve(in_mat, rhs):

    # in_mat must be 2x2 with nonzero determinant

    det_inv = 1.0/(in_mat[0][0]*in_mat[1][1] - in_mat[0][1]*in_mat[1][0])
    soln = [[None], [None]]
    soln[0][0] = det_inv*(in_mat[1][1]*rhs[0][0] - in_mat[0][1]*rhs[1][0])
    soln[1][0] = det_inv*(in_mat[0][0]*rhs[1][0] - in_mat[1][0]*rhs[0][0])

    return soln
########################################################################################################################
def svd(in_mat):

    # Assume in_mat is 2x2, exact arithmetic for computations, and that the singular values of on_mat are distinct
    V, rsvals = (in_mat.T*in_mat).diagonalize()
    V_norms = [V.col(0).norm(), V.col(1).norm()]

    for c in range(2):
        V[:, c] /= V_norms[c]

    V = sympy.MutableDenseMatrix(sympy.polys.polytools.cancel(V))

    U = sympy.MutableDenseMatrix(in_mat*V)

    singular_vals = [U.col(0).norm(), U.col(1).norm()]
    singular_vals = map(sympy.polys.polytools.cancel, singular_vals)

    if singular_vals[1] > singular_vals[0]:

        singular_vals = [singular_vals[1], singular_vals[0]]

        V0 = V.col(0)
        V[:, 0] = V[:, 1]
        V[:, 1] = V0

        U0 = U.col(0)
        U[:, 0] = U[:, 1]
        U[:, 1] = U0

    S = sympy.Matrix([[singular_vals[0], sympy.S.Zero], [sympy.S.Zero, singular_vals[1]]])

    for c in range(2):
        U[:, c] /= singular_vals[c]

    # diff = in_mat - U*S*V.T
    return sympy.polys.polytools.cancel(U), S, V
########################################################################################################################
def sing_vectors_plot(U, S, V):

    U1_x = [0.0, S[0, 0]*sympy.N(U[0, 0])]
    U1_y = [0.0, S[0, 0]*sympy.N(U[1, 0])]

    U2_x = [0.0, S[1, 1]*sympy.N(U[0, 1])]
    U2_y = [0.0, S[1, 1]*sympy.N(U[1, 1])]

    # U1_x = [0.0, sympy.N(U[0, 0])]
    # U1_y = [0.0, sympy.N(U[1, 0])]
    #
    # U2_x = [0.0, sympy.N(U[0, 1])]
    # U2_y = [0.0, sympy.N(U[1, 1])]

    V1_x = [0.0, sympy.N(V[0, 0])]
    V1_y = [0.0, sympy.N(V[1, 0])]

    V2_x = [0.0, sympy.N(V[0, 1])]
    V2_y = [0.0, sympy.N(V[1, 1])]

    matplotlib.pyplot.plot(U1_x, U1_y, 'k-')
    matplotlib.pyplot.plot(U2_x, U2_y, 'k-')

    matplotlib.pyplot.plot(V1_x, V1_y, 'r-')
    matplotlib.pyplot.plot(V2_x, V2_y, 'r-')


    matplotlib.pyplot.annotate('$U^1$', (U1_x[1], U1_y[1]))
    matplotlib.pyplot.annotate('$U^2$', (U2_x[1], U2_y[1]))

    matplotlib.pyplot.annotate('$V^1$', (V1_x[1], V1_y[1]))
    matplotlib.pyplot.annotate('$V^2$', (V2_x[1], V2_y[1]))

    matplotlib.pyplot.axes().set_aspect('equal')
    pass

########################################################################################################################
def unit_circle_plot(U, S, V, n):

    step = 2.0*math.pi/n
    x1 = [None]*(n+1)
    x2 = [None]*(n+1)

    y1 = [None]*(n+1)
    y2 = [None]*(n+1)

    for k in xrange(n):
        angle = k*step
        x1[k] = math.cos(angle)
        x2[k] = math.sin(angle)

        tmp = U*(S*(V.T*sympy.Matrix([x1[k], x2[k]])))
        y1[k] = tmp[0, 0]
        y2[k] = tmp[1, 0]

    x1[-1] = x1[0]
    x2[-1] = x2[0]

    y1[-1] = y1[0]
    y2[-1] = y2[0]

    matplotlib.pyplot.plot(x1, x2, 'r-')
    matplotlib.pyplot.plot(y1, y2, 'k-')

    pass

########################################################################################################################
def LU(A):

    # If A is not 2x2 and invertible, your face may explode.
    # P*A = L*U

    if (abs(A[0][1]) > abs(A[0][0])):
        # swap rows of A
        A = [[A[1][0], A[1][1]], [A[0][0], A[0][1]]]
        P = [[0][1], [1][0]]
    else:
        P = [[1, 0], [0, 1]]

    L = [[1, 0], [A[1][0]/A[0][0], 1]]
    U = [[A[0][0], A[0][1]], [0, A[1][1]-L[1][0]*A[0][1]]]

    return P, L, U
########################################################################################################################
def LU_solve(P, L, U, b):

    if P[0][0] != 1:
        # 0,0 entry of 2x2 permutation matrix indicates if effect of P is to swap rows
        b = [b[1], b[0]]

    y = [b[0], [b[1][0] - L[1][0]*b[0][0]]]

    x = [[None], [y[1][0]/U[1][1]]]
    x[0][0] = (y[0][0] - U[0][1]*x[1][0])/U[0][0]

    return x, y
########################################################################################################################
def comp_rel_err(x, x_hat):
    # x_vec = sympy.Matrix(x)
    x = (r[0] for r in x)
    x = map(sympy.Rational, x)
    x_vec = sympy.Matrix(x)
    abs_err = x_vec - sympy.Matrix(x_hat)
    return abs_err/x_vec.norm()
########################################################################################################################
def main():

    UU = sympy.zeros(2, 2)
    VV = sympy.zeros(2, 2)
    SS = sympy.zeros(2, 2)

    UU[0, 0] = sympy.S.Half*sympy.sqrt(3)
    UU[1, 0] = sympy.S.Half
    UU[0, 1] = -UU[1, 0]
    UU[1, 1] = UU[0, 0]

    SS[0, 0] = sympy.S.Half
    SS[1, 1] = sympy.S.One/1000000000

    VV[0, 0] = sympy.sqrt(2)/2
    VV[1, 0] = sympy.sqrt(2)/2
    VV[0, 1] = -VV[1, 0]
    VV[1, 1] = VV[0, 0]

    A = UU*SS*VV.T
    A_float = [[0.0, 0.0], [0.0, 0.0]]
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            A_float[r][c] = float(A[r, c])

    PA, LA, UA = LU(A_float)
    PA_sympy = []
    LA_sympy = []
    UA_sympy = []
    for r in range(2):
        PA_sympy.append(map(sympy.Rational, PA[r]))
        LA_sympy.append(map(sympy.Rational, LA[r]))
        UA_sympy.append(map(sympy.Rational, UA[r]))

    # D = sympy_mat(PA)*A - sympy_mat(LA)*sympy_mat(UA)
    # b = [[.86], [.5]]
    # x = LU_solve(PA, LA, UA, b)
    # r = sympy_mat(b) - sympy_mat(A_float)*sympy_mat(x)

    rhss = unit_circle(16)
    # rhss.insert(0, [[float(U[0, 0])], [float(U[1, 0])]])
    rhss_sympy = []
    float_solns = []
    float_y_solns = []
    for rhs in rhss:

        # Compute approximate solution in floating point
        # float_solns.append(det_solve(A_float, rhs))
        x, y = LU_solve(PA, LA, UA, rhs)
        float_solns.append(x)
        float_y_solns.append(y)

        rhss_sympy.append(copy.deepcopy(rhs))
        for r in range(2):
            rhss_sympy[-1][r][0] = sympy.Rational(rhss_sympy[-1][r][0])

        x_sympy, y_sympy = LU_solve(PA_sympy, LA_sympy, UA_sympy, rhss_sympy[-1])

        print sympy.N(comp_rel_err(x, x_sympy)), sympy.N(comp_rel_err(y, y_sympy))

    A = sympy_mat(A_float)
    A_inv = A.inv()

    U, S, V = svd(sympy.Matrix(A))
    S_inv = S.inv()


    # eps = 1.5e-9
    # float_mat = [[1.1 + eps, 1.1], [-2.2, -2.2 - eps]]
    # exact_mat = sympy.Matrix(sympy_mat(float_mat))
    # exact_mat_inv = exact_mat.inv()
    #
    # # U, S, V = svd(exact_mat)

    # sing_vectors_plot(UU, SS, VV)
    # unit_circle_plot(UU, SS, VV, 32)
    # matplotlib.pyplot.show()

    # SS_inv = SS.inv()
    # cond2 = SS[0, 0]/SS[1, 1]


    solns = []
    pert_rhss = []
    rhs_errs = []
    soln_errs = []
    for k in xrange(len(rhss)):

        rhss[k] = sympy_mat(rhss[k])
        float_solns[k] = sympy_mat(float_solns[k])

        solns.append(A_inv*rhss[k])

        # Compute rhs such that pert_soln is an exact solution to A*pert_soln = pert_rhs
        pert_rhss.append(A*float_solns[k])

        rhs_errs.append(rhss[k] - pert_rhss[k])

        soln_errs.append(solns[k] - float_solns[k])

    err_rhs_U_coords = []
    err_soln_V_coords = []
    err_rhs_U_coords_sing_inv = []
    err_comp_rhs_U_coords = []
    err_comp_soln_V_coords = []
    for k in xrange(len(rhss)):
        err_rhs_U_coords.append(U.T*(rhss[k] - pert_rhss[k]))
        err_rhs_U_coords_sing_inv.append(S_inv*err_rhs_U_coords[-1])
        err_soln_V_coords.append(V.T*(solns[k] - float_solns[k]))

        soln_V_coords = V.T*solns[k]
        rhs_U_coords = U.T*rhss[k]
        err_comp_rhs_U_coords.append(sympy.zeros(rhss[k].shape[0], 1))
        err_comp_soln_V_coords.append(sympy.zeros(rhss[k].shape[0], 1))
        for r in range(solns[k].shape[0]):
            err_comp_rhs_U_coords[-1][r] = err_rhs_U_coords[-1][r]/rhs_U_coords[r]
            err_comp_soln_V_coords[-1][r] = err_soln_V_coords[-1][r]/soln_V_coords[r]



    acc = 32
    for k in xrange(len(rhss)):
        print k, '*****************************'
        print 'soln'
        print sympy.N(solns[k])
        print 'soln abs err V coords'
        print sympy.N(err_soln_V_coords[k])
        print 'rhs abs err U coords'
        print sympy.N(err_rhs_U_coords[k])

    acc = 5
    for k in xrange(len(rhss)):
        # print 'rhs 2 norm rel err'
        # print sympy.N(rhs_errs[k].norm()/rhss[k].norm(), acc)
        #
        # print 'soln 2 norm rel err'
        # print sympy.N(soln_errs[k].norm()/solns[k].norm(), acc)

        print 'rhs scaled error U coords'
        tmp_rhs_rel_err_coords = sympy.N(err_comp_rhs_U_coords[k]/rhss[k].norm(), acc)
        print tmp_rhs_rel_err_coords

        print 'soln scaled error V coords'
        tmp_soln_rel_err_coords = sympy.N(err_soln_V_coords[k]/solns[k].norm(), acc)
        print tmp_soln_rel_err_coords

        # print 'rhs U residual'
        # print sympy.N(err_rhs_U_coords[k], acc)
        #
        # print sympy.N(S_inv*err_rhs_U_coords[k]/solns[k].norm(), acc)
        # print err_rhs_U_coords[k]
        print '****************'

        soln_abs_err_V_coords = V.T*soln_errs[k]
        soln_V_coords = V.T*solns[k]
        soln_rel_err_V_coords = soln_abs_err_V_coords
        for r in range(2):
            soln_rel_err_V_coords[r, 0] /= soln_V_coords[r, 0]

        rhs_U_coords = U.T*rhss[k]
        rhs_rel_err_U_coords = sympy.MutableDenseMatrix(err_rhs_U_coords[k])
        for r in range(2):
            rhs_rel_err_U_coords[r, 0] /= rhs_U_coords[r, 0]


        print 'cond2 lower bound'
        tmp_div = sympy.N(tmp_rhs_rel_err_coords.norm())
        if tmp_div == 0.0:
            print 'zero RHS error'
        else:
            print sympy.N(tmp_soln_rel_err_coords.norm())/tmp_div
        # print 'soln V coords component residual'
        # print sympy.N(soln_rel_err_V_coords, acc)
        # print 'rhs U coords component residual'
        # print sympy.N(rhs_rel_err_U_coords, acc)


########################################################################################################################
if __name__ == '__main__':
    main()
