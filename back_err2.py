import argparse
import sys

import numpy.linalg
import pickle
import sympy
import float_solve
import exact_solve


class CommandLineArgs(object):

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_angles",
                        "-n",
                        required=True,
                        type=int,
                        help="Split the perimeter unit circle into this many evenly divided lengths.")

    parser.add_argument("--singular_vals",
                        "-s",
                        required=True,
                        help="Space separated list of singular values.")


def main():

    cl_args = CommandLineArgs.parser.parse_args(sys.argv[1:])

    ###################################################################################
    U_angle = numpy.pi/3.0
    U_trig = [numpy.cos(U_angle), numpy.sin(U_angle)]
    V_angle = numpy.pi/4.0
    V_trig = [numpy.cos(V_angle), numpy.sin(V_angle)]

    U = rotation_matrix(*U_trig)
    V = rotation_matrix(*V_trig)

    try:

        singular_values = [float(sv) for sv in cl_args.singular_vals.split()]

    except ValueError:

        sys.stderr.write("Invalid command line singular values:%s\n")
        sys.exit(1)

    S = numpy.diagflat(singular_values)

    A = numpy.dot(U, S)
    A = numpy.dot(A, V.T)

    # Solve A*x = b in double precision for equispaced right hand sides on the unit circle
    num_angles = cl_args.num_angles
    angle_factor = 2.0*numpy.pi/num_angles

    rhs_angles = [k*angle_factor for k in range(num_angles)]
    dbl_rhss = [numpy.array([[numpy.cos(angle)], [numpy.sin(angle)]]) for angle in rhs_angles]

    # Compute solutions to perturbed problem: xhat = solve(A, b), in floating point
    dbl_A_solver = float_solve.PLUsolve(A)
    dbl_perturbed_solns = map(dbl_A_solver, dbl_rhss)

    # Compute EXACT solutions to A*x=b: x = inv(A)*b, in exact arithmetic
    exact_A_solver = exact_solve.solve(A)
    exact_solns = map(exact_A_solver, dbl_rhss)

    # Compute perturbed right hand sides: bhat = A*xhat, in exact arithmetic
    calculate_perturbed_rhs = exact_solve.perturbed_RHS(A)
    perturbd_rhss = map(calculate_perturbed_rhs, dbl_perturbed_solns)

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


def rotation_matrix(cosine, sine):
    return numpy.array([[cosine, -sine], [sine, cosine]])


if __name__ == '__main__':
    main()