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

    U = sympy.polys.polytools.cancel(U)
    # diff = in_mat - U*S*V.T
    return U, S, V
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
        P = [[0, 1], [1, 0]]
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
def rotation_matrix(angle):

    tmp_vec = sympy.Matrix([sympy.cos(angle), sympy.sin(angle)])
    tmp_vec /= tmp_vec.norm()

    row1 = [sympy.cos(angle), -sympy.sin(angle)]
    row2 = [-row1[1], row1[0]]
    return sympy.Matrix([row1, row2])

########################################################################################################################
def matrix_to_list(in_mat):
    out_list = []
    for r in range(in_mat.shape[0]):
        out_list.append(list(in_mat.row(r)))
    return out_list
########################################################################################################################
def main():

    theta_u = sympy.S.Pi/6
    theta_v = sympy.S.Pi/2

    UU = rotation_matrix(theta_u)
    VV = rotation_matrix(theta_v)

    SS = sympy.zeros(2, 2)
    SS[0, 0] = sympy.S.One
    SS[1, 1] = sympy.S.One/10**8

    A_init = UU*SS*VV.T
    A = [[0.0, 0.0], [0.0, 0.0]]
    for r in range(len(A)):
        for c in range(len(A[1])):
            A[r][c] = float(A_init[r, c])

    P_float, L_float, U_float = LU(A)

    # Matrix of Sympy Rationals
    A_rational = sympy_mat(A)
    A_rational = [list(A_rational.row(k)) for k in range(2)]

    P_rational, L_rational, U_rational = LU(A_rational)

    num_angles = 8
    angles = [float(k)/float(num_angles)*2*math.pi for k in range(num_angles)]
    bs = [[[math.cos(angle)], [math.sin(angle)]] for angle in angles]
    xhats = []
    yhats = []
    bhats = []

    xs = []
    ys = []

    err = [[None], [None]]
    soln_abs_errs = []
    soln_rel_errs = []
    back_abs_errs = []
    back_rel_errs = []

    for b in bs:
        xhat, yhat = LU_solve(P_float, L_float, U_float, b)
        xhats.append(xhat)
        yhats.append(yhat)

        # Exact solution
        b_rational = [[sympy.Rational(b[0][0])], [sympy.Rational(b[1][0])]]
        x, y = LU_solve(P_rational, L_rational, U_rational, b_rational)
        xs.append(x)
        ys.append(y)

        # Exact right hand side
        wuuut
        x_hat_rational = [[sympy.Rational(xhat[0][0])], [sympy.Rational(xhat[1][0])]]
        bhat = sympy.Matrix(A_rational)*sympy.Matrix(x_hat_rational)
        bhat = matrix_to_list(bhat)
        bhats.append(bhat)

        # forward error
        for r in range(2):
            err[r][0] = x[r][0] - sympy.Rational(xhat[r][0])

        soln_abs_errs.append(err)
        soln_rel_errs.append(sympy.Matrix(err).norm() / sympy.Matrix(x).norm())

        # backwards error
        for r in range(2):
            err[r][0] = b_rational[r][0] - bhat[r][0]

    return
    U, S, V = svd(sympy.Matrix(A_rational))

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

    b_floats = unit_circle(16)
    # rhss.insert(0, [[float(U[0, 0])], [float(U[1, 0])]])
    rhss_sympy = []
    x_hat_floats = []
    y_hat_floats = []
    b_hats = []
    for b_float in b_floats:

        # Compute approximate solution in floating point
        # float_solns.append(det_solve(A_float, rhs))
        x_hat_float, y_hat_float = LU_solve(PA, LA, UA, b_float)
        x_hat_floats.append(x_hat_float)
        y_hat_floats.append(y_hat_float)

        rhss_sympy.append(copy.deepcopy(b_float))
        for r in range(2):
            rhss_sympy[-1][r][0] = sympy.Rational(rhss_sympy[-1][r][0])

        b_hats.append(A_sympy*sympy_mat(x_hat_float))

        b_tmp = sympy_mat(b_float)
        b_comp_rel_err = (b_hats[-1] - b_tmp)/b_tmp.norm()
        print sympy.N(b_comp_rel_err)


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
    for k in xrange(len(b_floats)):

        b_floats[k] = sympy_mat(b_floats[k])
        x_hat_floats[k] = sympy_mat(x_hat_floats[k])

        solns.append(A_inv*b_floats[k])

        # Compute rhs such that pert_soln is an exact solution to A*pert_soln = pert_rhs
        pert_rhss.append(A*x_hat_floats[k])

        rhs_errs.append(b_floats[k] - pert_rhss[k])

        soln_errs.append(solns[k] - x_hat_floats[k])

    err_rhs_U_coords = []
    err_soln_V_coords = []
    err_rhs_U_coords_sing_inv = []
    err_comp_rhs_U_coords = []
    err_comp_soln_V_coords = []
    for k in xrange(len(b_floats)):
        err_rhs_U_coords.append(U.T*(b_floats[k] - pert_rhss[k]))
        err_rhs_U_coords_sing_inv.append(S_inv*err_rhs_U_coords[-1])
        err_soln_V_coords.append(V.T*(solns[k] - x_hat_floats[k]))

        soln_V_coords = V.T*solns[k]
        rhs_U_coords = U.T*b_floats[k]
        err_comp_rhs_U_coords.append(sympy.zeros(b_floats[k].shape[0], 1))
        err_comp_soln_V_coords.append(sympy.zeros(b_floats[k].shape[0], 1))
        for r in range(solns[k].shape[0]):
            err_comp_rhs_U_coords[-1][r] = err_rhs_U_coords[-1][r]/rhs_U_coords[r]
            err_comp_soln_V_coords[-1][r] = err_soln_V_coords[-1][r]/soln_V_coords[r]



    acc = 32
    for k in xrange(len(b_floats)):
        print k, '*****************************'
        print 'soln'
        print sympy.N(solns[k])
        print 'soln abs err V coords'
        print sympy.N(err_soln_V_coords[k])
        print 'rhs abs err U coords'
        print sympy.N(err_rhs_U_coords[k])

    acc = 5
    for k in xrange(len(b_floats)):
        # print 'rhs 2 norm rel err'
        # print sympy.N(rhs_errs[k].norm()/rhss[k].norm(), acc)
        #
        # print 'soln 2 norm rel err'
        # print sympy.N(soln_errs[k].norm()/solns[k].norm(), acc)

        print 'rhs scaled error U coords'
        tmp_rhs_rel_err_coords = sympy.N(err_comp_rhs_U_coords[k]/b_floats[k].norm(), acc)
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

        rhs_U_coords = U.T*b_floats[k]
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
