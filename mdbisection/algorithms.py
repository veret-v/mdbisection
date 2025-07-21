from __future__ import annotations

import copy

import numpy as np

from mdbisection.simplex import Simplex
from mdbisection.simplex_system import StandartSimplex, SimplexSystem


def top_deg_stage(
        working_simplex : Simplex, 
        simplexes       : list,
        func : function, 
        max_diameter : float, 
        epsilon1 : float, 
        m : int, 
        max_num_approx : int,
        dim : float
    ) -> tuple:
    """
    Purpose: Handles the topological degree computation stage of the bisection algorithm when either:
    1. The method fails to converge within maximum iterations, or
    2. Standard simplex containment checks prove insufficient
    This safeguards convergence by reverting to topological analysis.

    :param_data: working_simplex: Current active simplex being refined (Simplex object)
    :param_data: simplexes: List of all active simplices in the search tree (list[Simplex])
    :param_data: func: Target function f: ℝⁿ → ℝⁿ defining the system (callable)
    :param_data: max_diameter: Initial diameter of the search domain (float)
    :param_data: epsilon1: Spatial tolerance threshold for simplex size (float)
    :param_data: m: Current iteration counter (int)
    :param_data: max_num_approx: Maximum allowed iterations before degree check (int)
    :param_data: dim: Problem dimension n for ℝⁿ (int)

    :return: tuple containing:
        - simplex: Verified simplex with non-zero topological degree
        - m: Reset iteration counter
    """
    if working_simplex.calc_diameter() / max_diameter <= 10 * epsilon1:
        if working_simplex.calc_topological_degree(func) != 0:
            m = 0
            simplexes.clear()
            simplexes.append(working_simplex)
            return working_simplex, m
    
    j = m
    while (j > 0):
        simplex1, simplex2 = simplexes[j].bisect()

        if simplex1.calc_topological_degree(func) != 0:
            m = 0
            simplexes.clear()
            simplexes.append(simplex1)
            return simplex1, m
        
        if simplex2.calc_topological_degree(func) != 0:
            m = 0 
            simplexes.clear()
            simplexes.append(simplex2)
            return simplex2, m
        
        j -= 1

    simplex1, simplex2 = simplexes[0].bisect()
    m = 0
    simplexes.clear()

    if simplex1.calc_topological_degree(func) != 0:
        simplexes.append(simplex1)
        return simplex1, m
    simplexes.append(simplex2)
    return simplex2, m

    
def solve_det_system(
        func : function, 
        init_simplex : Simplex, 
        epsilon1 : float, 
        epsilon2 : float, 
        dim : int,
        max_num_approx : int

    ) -> np.ndarray:
    """
    Purpose: Finds solutions to nonlinear equation systems using generalized multidimensional bisection method.
            Implements a robust algorithm that combines simplex bisection with topological degree computation
            to guarantee convergence to roots within the specified domain.

    :param_data: func: The vector function f: ℝⁿ → ℝⁿ defining the system of equations to solve
    :param_data: init_simplex: Initial simplex defining the search domain in ℝⁿ⁺¹ space
    :param_data: epsilon1: Tolerance for simplex diameter (stopping criterion for spatial refinement)
    :param_data: epsilon2: Tolerance for function values (stopping criterion for solution accuracy) 
    :param_data: dim: Dimension of the problem (n for ℝⁿ)
    :param_data: max_num_approx: Maximum number of simplex refinements before reverting 
                                to topological degree verification stage

    :return: tuple containing:
            - simplexes: List of all convergent simplices that potentially contain solutions
            - solution: Approximate solution vector when found (None if no solution guaranteed)
    """
    m = 0
    max_diameter = init_simplex.calc_diameter()
    simplexes = []
    output_simplexes = []
    working_simplex = init_simplex
    simplexes.append(copy.deepcopy(working_simplex))
    output_simplexes.append(copy.deepcopy(working_simplex))

    if init_simplex.calc_topological_degree(func) == 0:
        print("Нет гарантии существования корня в симплексе. Метод не может быть применен.")
        return simplexes, None

    while True:
        if (working_simplex.calc_diameter() < epsilon1):
            affine_trans = []
            for i in range(dim):
                coeffs = np.vstack(
                    [
                        np.stack([point for point in working_simplex.points]).T, 
                        np.ones(dim + 1)
                    ]
                )
                affine_row = np.linalg.solve(
                    coeffs.T, 
                    np.array([func(point)[i] for point in working_simplex.points])
                )
                affine_trans.append(affine_row)
            affine_trans = np.stack(affine_trans) 
            approx_solution = np.linalg.solve(affine_trans[:, :-1], -affine_trans[:, -1])
            return output_simplexes, approx_solution
        
        if (working_simplex.calc_point_norms(func) < epsilon2).any():
            approx_solution = np.array(working_simplex.points)[
                (working_simplex.calc_point_norms(func) < epsilon2)
            ][0]
            return output_simplexes, approx_solution
        
        if m == max_num_approx:
            working_simplex, m = top_deg_stage(
                working_simplex, simplexes, func, 
                max_diameter, epsilon1, m, max_num_approx, dim
            )
            max_diameter = working_simplex.calc_diameter()

        else:
            simplex1, simplex2 = working_simplex.bisect()

            if simplex1.transform(func).check_point(np.zeros(dim)):
                m += 1
                simplexes.append(simplex1)
                working_simplex = simplex1

            elif simplex2.transform(func).check_point(np.zeros(dim)):
                m += 1
                simplexes.append(simplex2)
                working_simplex = simplex2

            else:
                refl_simplex1, refl_simplex2, flag = working_simplex.reflection(func)

                if refl_simplex1.transform(func).check_point(np.zeros(dim)) and flag:
                    m += 1
                    simplexes.append(refl_simplex1)
                    working_simplex = refl_simplex1

                elif refl_simplex2.transform(func).check_point(np.zeros(dim)) and flag:
                    m += 1
                    simplexes.append(refl_simplex2)
                    working_simplex = refl_simplex2

                else:
                    working_simplex, m = top_deg_stage(
                        working_simplex, simplexes, func, 
                        max_diameter, epsilon1, m, max_num_approx, dim
                    )
                    max_diameter = working_simplex.calc_diameter()

        output_simplexes.append(working_simplex)   


def optimize(
        M : float, 
        func : function, 
        dim : int, 
        center : np.ndarray, 
        r : float, 
        epsilon : float, 
        initial_simplex : StandartSimplex
    ) -> tuple:
    """
    Purpose: Optimizes a multivariable function using the multidimensional bisection method
             within a specified domain by iteratively reducing simplex systems
    
    :param_data: M: Lipschitz constant for the function
    :param_data: func: The target function f: R^n → R to be optimized
    :param_data: dim: Dimension of the problem space
    :param_data: center: Central point of the initial search domain
    :param_data: r: Radius of the initial search domain
    :param_data: epsilon: Convergence threshold for stopping the algorithm
    :param_data: initial_simplex: Starting simplex for the optimization
    
    :return: tuple: (list of all evaluated simplexes, coordinates of the found optimum)
    """
    # initial_simplex = gen_init_simplex(r, center, M, func, dim)
    lowest_top = initial_simplex.apex.copy()
    lowest_top[-1] = lowest_top[-1] + initial_simplex.height
    working_system = SimplexSystem([initial_simplex], dim + 1, dim + 2, lowest_top)
    out_simplexes = []

    while working_system.variation > epsilon:
        working_system = working_system.reduction(func).elimination()
        out_simplexes.append(working_system.simplexes[0])
        print(working_system.variation)
        print(len(working_system.simplexes))
        
    return out_simplexes, working_system.lowest_top


def gen_init_simplex(r : float, c : np.ndarray, M : float, func : function, dim : int) -> StandartSimplex:
    """
    Purpose: Generates an initial standard simplex for the optimization algorithm
             based on function evaluations at strategic points
    
    :param_data: r: Radius of the search domain
    :param_data: c: Center point of the search domain
    :param_data: M: Lipschitz constant for the function
    :param_data: func: The target function f: R^n → R to be optimized
    :param_data: dim: Dimension of the problem space
    
    :return: StandartSimplex: Properly configured initial simplex for optimization
    """
    u_k = np.zeros((dim + 2, dim + 1))
    for k in range(dim + 1):
        u_k[k, k] = 1.0
    u_k[-1, :] = (-1.0)/(1 + np.sqrt(dim + 2)) * np.ones(dim + 1)
    for k in range(dim + 1):
        u_k[k, :] = u_k[k, :] - u_k[-1, :]
    
    u_k = u_k / np.linalg.norm(u_k[0])
    
    v = np.array([c.coords - r * u_k for u_k in u_k])
    f_values  = np.array([func(v_k) for v_k in v])
    m = np.min(f_values)
    
    sum_f = np.sum(f_values)
    sum_terms = 0
    for k in range(dim + 1):
        sum_terms += (f_values[k] - m) * u_k[k]
    
    apex = c.coords + (1.0 / (M *  (dim + 2))) * sum_terms
    apex[dim] = (sum_f/(dim + 2)) - M * (dim + 1)
    apex = apex
    height = M * dim * r - (1.0/(dim + 2)) * (sum_f - (dim + 2) * m) 
    
    base_vertices = []
    for k in range(dim + 1):
        vertex = apex.coords.copy()
        vertex[:dim] += (height / M) * u_k[k][:dim]
        vertex[dim] += height
        base_vertices.append(vertex)
    
    return StandartSimplex(base_vertices + [apex], dim + 1, dim + 2)
