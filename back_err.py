import scipy.linalg
import numpy.linalg
import numpy
import sympy
import matplotlib
import matplotlib.pyplot
import matplotlib.ticker
import mpl_toolkits.mplot3d.axes3d

class Errors(object):

    def __init__(self, soln_diffs, exact_solns, rhs_diffs, dbl_rhss):

        self.soln_diffs = soln_diffs
        self.exact_solns = exact_solns
        self.rhs_diffs = rhs_diffs
        self.rhss = [sympy.Matrix(map(sympy.Rational, (rhs[0][0], rhs[1][0]))) for rhs in dbl_rhss]

        soln_rel_errs = [soln_diff.norm()/soln.norm() for soln_diff, soln in zip(self.soln_diffs, self.exact_solns)]
        rhs_rel_errs = [rhs_diff.norm()/rhs.norm() for rhs_diff, rhs in zip(self.rhs_diffs, self.rhss)]
        cond_lowers = [soln_rel_err/rhs_rel_err for soln_rel_err, rhs_rel_err in zip(soln_rel_errs, rhs_rel_errs)]

        self.soln_rel_errs = [sympy.N(soln_rel_err) for soln_rel_err in soln_rel_errs]
        self.rhs_rel_errs = [sympy.N(rhs_rel_err) for rhs_rel_err in rhs_rel_errs]
        self.cond_lowers = [sympy.N(cond_lower) for cond_lower in cond_lowers]

def convert_numpy_array(numpy_array):
    return [(arr[0][0], arr[1][0]) for arr in numpy_array]


def log_tick_formatter(val, pos=None):
    # Matplotlib's implementation for log scale axes in 3D plots doesn't appear to work:
    # https://github.com/matplotlib/matplotlib/issues/209
    return "{:.1e}".format(10**val)

def plot_error(x, y, z):

    fig = matplotlib.pyplot.figure()
    ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig)

    ax.plot(x, y, z, "-o", markersize=3)
    # ax.scatter(x, y, z)

    ticks = [-1, -.5, .0, .5, 1]
    str_ticks = map(str, ticks)

    ax.set_xticks(ticks)
    ax.set_xticklabels(str_ticks)
    ax.set_xlabel("x")

    ax.set_yticks(ticks)
    ax.set_yticklabels(str_ticks)
    ax.set_ylabel("y")

    return fig

def PLUsolve(A):
    """
    Return solver suitable for A*x=b for map(), where the approximate solution is computed in
    floating point arithmetic by scipy's wrapper for LAPACK DGETRF.
    Returned solver maps right hand side b to soln xhat for fixed coefficient matrix A.
    :param A:
    :return:
    """

    # Compute L U factorization of A via wrapper of LAPACK DGETRF
    LU, P = scipy.linalg.lu_factor(A)

    def rhs_solver(b):
        return scipy.linalg.lu_solve((LU, P), b)

    return rhs_solver

def QRsolve(A):

    Q, R = numpy.linalg.qr(A)

    def rhs_solver(b):
        return scipy.linalg.solve_triangular(R, numpy.dot(Q.T, b))

    return rhs_solver

def exact_A(A):
    """
    Return a sympy.Matrix instance built from a matrix of floating point numbers.
    The entries in the returned matrix are sympy.Rationals that exactly match the entries of
    the input matrix.
    :param A:
    :return:
    """

    return sympy.Matrix([map(sympy.Rational, row) for row in A])

def exact_solve(dbl_A):
    """
    Return solver for A*x=b suitable for map(), where the exact solution is computed.
    :param dbl_A:
    :return: Callable f such that f(b) returns the exact solution to dbl_A*x=b
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

def rotation_matrix(cosine, sine):
    """
    Helper function for constructing U and V factors in singular value decomposition A=U*Sigma*V.Transpose
    :param cosine:
    :param sine:
    :return:
    """
    return numpy.array([[cosine, -sine], [sine, cosine]])


def main():

    # Construct A in floating point arithmetic from its prescribed singular value decomposition.
    U_angle = numpy.pi/3.0
    U_trig = [numpy.cos(U_angle), numpy.sin(U_angle)]
    V_angle = numpy.pi/4.0
    V_trig = [numpy.cos(V_angle), numpy.sin(V_angle)]

    U = rotation_matrix(*U_trig)
    V = rotation_matrix(*V_trig)

    singular_values = [1, 2**-24]
    S = numpy.diagflat(singular_values)

    A = numpy.dot(U, S)
    A = numpy.dot(A, V.T)

    # Solve A*x = b in double precision for equispaced right hand sides on the unit circle
    num_angles = 1024
    angle_factor = 2.0*numpy.pi/num_angles

    rhs_angles = [k*angle_factor for k in range(num_angles)]
    dbl_rhss = [numpy.array([[numpy.cos(angle)], [numpy.sin(angle)]]) for angle in rhs_angles]

    # Compute solutions to perturbed problem: xhat = solve(A, b), in floating point
    # dbl_A_solver is a callable such that xhat = dbl_A_solver(dbl_rhs).

    # Solve by Gaussian elimination with partial pivoting, and two sequential triangular solves
    # dbl_A_solver = PLUsolve(A)
    dbl_A_solver = QRsolve(A)

    # Solve by QR
    dbl_perturbed_solns = map(dbl_A_solver, dbl_rhss)

    # Compute EXACT solutions to A*x=b: x = inv(A)*b, in exact arithmetic
    # exact_A returns a callable such that x
    exact_A_solver = exact_solve(A)
    exact_solns = map(exact_A_solver, dbl_rhss)

    # Compute perturbed right hand sides: bhat = A*xhat, in exact arithmetic
    calculate_perturbed_rhs = perturbed_RHS(A)
    perturbd_rhss = map(calculate_perturbed_rhs, dbl_perturbed_solns)

    # Compute perturbed right hand sides: bhat = A*xhat, in exact arithmetic
    rhs_diffs = [exact_A(exact)-perturbed for exact, perturbed in zip(dbl_rhss, perturbd_rhss)]
    soln_diffs = [-(exact - exact_A(perturbed)) for exact, perturbed in zip(exact_solns, dbl_perturbed_solns)]
    assert(len(soln_diffs) == len(rhs_diffs))

    # rhs_rel_errs = [rhs_abs_diff/exact_solve.exact_A(exact_rhs).norm() for rhs_abs_diff, exact_rhs in zip(rhs_diffs, dbl_rhss)]
    # soln_rel_errs = [soln_abs_diff/exact_soln.norm() for soln_abs_diff, exact_soln in zip(soln_diffs, exact_solns)]
    #
    # cond2_lower_bounds = [soln_rel_err.norm()/rhs_rel_err.norm()
    #                       for soln_rel_err, rhs_rel_err in zip(soln_rel_errs, rhs_rel_errs)]
    #
    # print [sympy.N(cond2_lower_bound) for cond2_lower_bound in cond2_lower_bounds]

    # U, D, V = exact_solve.svd(exact_solve.exact_A(A))

    errors = Errors(soln_diffs, exact_solns, rhs_diffs, dbl_rhss)
    tmp_rhss = convert_numpy_array(dbl_rhss)

    rhs_x = [rhs[0] for rhs in tmp_rhss]
    rhs_x.append(rhs_x[0])
    rhs_x = map(float, rhs_x)
    rhs_y = [rhs[1] for rhs in tmp_rhss]
    rhs_y.append(rhs_y[0])
    rhs_y = map(float, rhs_y)

    # Solution relative error
    z = errors.soln_rel_errs
    z.append(z[0])
    z = map(float, z)
    z = numpy.log10(z)

    fig = plot_error(rhs_x, rhs_y, z)
    axes = fig.get_axes()[0]
    # https://github.com/matplotlib/matplotlib/issues/209
    axes.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(log_tick_formatter))
    axes.set_label("Solution relative 2-norm error")
    fig.savefig("/tmp/soln_rel_err.png", format="png")
    # print axes.get_zlim()
    # zlabels = axes.get_zticklabels()
    # zticks = axes.get_zticks()
    # axes.set_zlim(.9*min(map(float, errors.soln_rel_errs)), 1.1*max(map(float, errors.soln_rel_errs)))
    # axes.set_zlim(axes.get_zlim())
    # ticks = [1e-10, 1e-9]
    # axes.set_zticks(ticks)
    # axes.set_zticklabels(map(str, ticks))
    # matplotlib.pyplot.show()
    # fig.savefig("/tmp/soln_rel_err2.png", format="png")
    # zlabels = axes.get_zticklabels()
    # zticks = axes.get_zticks()
    # print "soln rel err plot 2 axes.get_zlim():", axes.get_zlim()

    print axes.get_zlim()
    print "soln rel err min max:", (min(map(float, errors.soln_rel_errs)), max(map(float, errors.soln_rel_errs)))


    # RHS relative error
    z = errors.rhs_rel_errs
    z.append(z[0])
    z = map(float, z)
    z = numpy.log10(z)

    fig = plot_error(rhs_x, rhs_y, z)
    axes = fig.get_axes()[0]
    axes.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(log_tick_formatter))

    # axes.set_zscale('log')
    # axes.set_zlim(.9*min(map(float, errors.rhs_rel_errs)), 1.1*max(map(float, errors.rhs_rel_errs)))
    # ticks = [1e-1, 1e2]
    # axes.set_zticks(ticks)
    # axes.set_zticklabels(map(str, ticks))
    axes.set_label("RHS relative 2-norm error")
    # matplotlib.pyplot.show()
    fig.savefig("/tmp/rhs_rel_err.png", format="png")
    zlabels = axes.get_zticklabels()
    zticks = axes.get_zticks()
    print "rhs rel err min max:", (min(map(float, errors.rhs_rel_errs)), max(map(float, errors.rhs_rel_errs)))

    # 2-norm condition number lower bound
    z = errors.cond_lowers
    z.append(z[0])
    z = map(float, z)
    z = numpy.log10(z)

    fig = plot_error(rhs_x, rhs_y, z)
    axes = fig.get_axes()[0]

    axes.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(log_tick_formatter))
    # axes.set_zlim(.9*min(map(float, errors.cond_lowers)), 1.1*max(map(float, errors.cond_lowers)))
    # ticks = [1e-1, 1, 2]
    # axes.set_zticks(ticks)
    # axes.set_zticklabels(map(str, ticks))
    axes.set_label("2-norm condition number lower bound")
    # matplotlib.pyplot.show()
    fig.savefig("/tmp/cond-lower.png", format="png")

    zlabels = axes.get_zticklabels()
    zticks = axes.get_zticks()
    print(zlabels)
    print "cond lower bound:", (min(map(float, errors.cond_lowers)), max(map(float, errors.cond_lowers)))

if __name__ == '__main__':
    main()