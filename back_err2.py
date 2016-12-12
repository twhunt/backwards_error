import numpy.linalg
import scipy.linalg
import sympy
import float_solve
import exact_solve


def main():


    ###################################################################################
    U_angle = numpy.pi/3.0
    U_trig = [numpy.cos(U_angle), numpy.sin(U_angle)]
    V_angle = numpy.pi/4.0
    V_trig = [numpy.cos(V_angle), numpy.sin(V_angle)]

    U = rotation_matrix(*U_trig)
    V = rotation_matrix(*V_trig)
    singular_values = [1, 2**-28]
    S = numpy.diagflat(singular_values)

    A = numpy.dot(U, S)
    A = numpy.dot(A, V.T)

    ###################################################################################
    # Solve A*x = b in double precision for equispaced right hand sides on the unit circle
    num_angles = 4
    angle_factor = 2.0*numpy.pi/num_angles

    rhs_angles = [k*angle_factor for k in range(num_angles)]
    dbl_rhss = [numpy.array([[numpy.cos(angle)], [numpy.sin(angle)]]) for angle in rhs_angles]

    ###################################################################################
    # Compute solutions to perturbed problem: xhat = solve(A, b), in floating point
    dbl_A_solver = float_solve.PLUsolve(A)
    dbl_perturbed_solns = map(dbl_A_solver, dbl_rhss)

    ###################################################################################
    # Compute EXACT solutions to A*x=b: x = inv(A)*b, in exact arithmetic arithmetic
    exact_A_solver = exact_solve.solve(A)
    exact_solns = map(exact_A_solver, dbl_rhss)

    ###################################################################################
    # Compute perturbed right hand sides: bhat = A*xhat, in exact arithmetic
    calculate_perturbed_rhs = exact_solve.perturbed_RHS(A)
    perturbd_rhss = map(calculate_perturbed_rhs, dbl_perturbed_solns)

    ###################################################################################
    # Compute perturbed right hand sides: bhat = A*xhat, in exact arithmetic
    rhs_diffs = [exact_solve.exact_A(exact)-perturbed for exact, perturbed in zip(dbl_rhss, perturbd_rhss)]
    soln_diffs = [-(exact - exact_solve.exact_A(perturbed)) for exact, perturbed in zip(exact_solns, dbl_perturbed_solns)]
    assert(len(soln_diffs) == len(rhs_diffs))

    rhs_rel_errs = [rhs_abs_diff/exact_solve.exact_A(exact_rhs).norm() for rhs_abs_diff, exact_rhs in zip(rhs_diffs, dbl_rhss)]
    soln_rel_errs = [soln_abs_diff/exact_soln.norm() for soln_abs_diff, exact_soln in zip(soln_diffs, exact_solns)]

    cond2_lower_bounds = [soln_rel_err.norm()/rhs_rel_err.norm()
                          for soln_rel_err, rhs_rel_err in zip(soln_rel_errs, rhs_rel_errs)]

    print [sympy.N(cond2_lower_bound) for cond2_lower_bound in cond2_lower_bounds]

    U, D, V = exact_solve.svd(exact_solve.exact_A(A))

    cond2 = sympy.simplify(D[0, 0]/D[1, 1])

    # ###################################################################################
    # # Compute RHS error coordinate vector
    # U_RHS_err_coord_vecs = [U.T*abs_diffs[k][1] for k in range(len(abs_diffs))]
    #
    # exact_solns = []
    # dbl_rhss_rational = []
    # soln_abs_errs = []
    # rhs_abs_errs = []
    #
    # perturbed_rhss_rational = []
    # # IN:
    # # rational A: exact A, represented as sympy.Rationals
    # # dbl_perturbed_solns: solutions to problem with exact A and perturbed RHS, represented as doubles
    # # dbl_rhss: exact right hand sides, represented as doubles
    #
    #
    # # OUT:
    # # Exact solution to unperturbed problem, and exact RHS to perturbed problem
    # # perturbed_rhss_rational: exact RHS to problem with perturbed solution
    # for dbl_soln, dbl_rhs in zip(dbl_perturbed_solns, dbl_rhss):
    #
    #     # Represent the solution to the problem with perturbed RHS with sympy.Rationals
    #     dbl_perturbed_soln_rational = sympy.Matrix([sympy.Rational(v[0]) for v in dbl_soln])
    #
    #     # Compute the perturbed RHS, given the exact A and solution to the problem with perturbed RHS
    #     perturbed_rhss_rational.append(rational_A*dbl_perturbed_soln_rational)
    #
    #     # Represent unperturbed RHS with sympy.Rationals
    #     dbl_rhss_rational.append(sympy.Matrix([sympy.Rational(v[0]) for v in dbl_rhs]))
    #
    #     # Compute exact solution to unperturbed problem
    #     exact_solns.append(svd_inv(U, D, V, dbl_rhss_rational[-1]))
    #
    #     exact_soln_rational = sympy.Matrix([sympy.Rational(v[0]) for v in dbl_soln])
    #     soln_abs_errs.append(exact_solns[-1] - exact_soln_rational)
    #
    #     rhs_abs_errs.append(dbl_rhss_rational[-1] - perturbed_rhss_rational[-1])
    #
    #     print sympy.N(U.T * (dbl_rhss_rational[0] - perturbed_rhss_rational[0]) / dbl_rhss_rational[0].norm())
    # pass

def rotation_matrix(cosine, sine):
    return numpy.array([[cosine, -sine], [sine, cosine]])


if __name__ == '__main__':
    main()