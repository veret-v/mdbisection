from __future__ import annotations

import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from mdbisection.simplex import Simplex
from mdbisection.simplex_system import StandartSimplex, SimplexSystem
from mdbisection.char_polyhedron import CharPolyhedron


def __check_level_lines(
        func : function,
        working_simplex : Simplex,
        max_iter: int = 8
    ) -> tuple:
    """
    Purpose: Handles the detection of level lines presence through iterative simplex bisection.
    Performs multiple bisection cycles and checks for sign pattern consistency across simplex vertices
    to identify regions containing solution curves. Provides fallback when standard convergence fails.
    
    :param func: Target function f: ℝⁿ → ℝⁿ defining the system (callable)
    :param working_simplex: Current active simplex being analyzed (Simplex object)
    :param max_iter: Maximum number of bisection iterations to perform (int)
    
    :return: tuple containing:
        - bool: True if level lines detected, False otherwise
        - Simplex: Refined simplex where level lines were detected or original simplex if not found
    """
    simplexes = [working_simplex]
    iteration = 0
    while(iteration < max_iter):
        new_simplexes = []
        for simplex in simplexes:
            simplex1, simplex2 = simplex.bisect()
            new_simplexes.append(simplex1)
            new_simplexes.append(simplex2)
        simplexes = new_simplexes.copy()

        for simplex in simplexes:
            sgn_matrix = Simplex.calc_sgn_matrix(func, simplex)
            column_sgn = np.all(sgn_matrix == sgn_matrix[0,:], axis = 0)
            if not column_sgn.any() and simplex.calc_topological_degree(func) != 0:
                return True, simplex
        iteration += 1
    return False, working_simplex
                

def __top_deg_stage(
        working_simplex : Simplex, 
        simplexes : list,
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
    while (j >= 0):
        scales = [0.3, 0.5, 0.7]

        for scale in scales:
            simplex1, simplex2 = simplexes[j].bisect(scale=scale)

            simplex1_in = simplex1.calc_topological_degree(func)
            simplex2_in = simplex2.calc_topological_degree(func)
                
            if simplex1_in != 0 and simplex2_in == 0:
                m = 0
                simplexes.clear()
                simplexes.append(simplex1)
                return simplex1, m
            
            elif simplex2_in != 0 and simplex1_in == 0:
                m = 0 
                simplexes.clear()
                simplexes.append(simplex2)
                return simplex2, m

        j -= 1
    return None, 0


def __topologic_deg_bisect(
        func : function, 
        init_simplex : Simplex, 
        epsilon1 : float, 
        epsilon2 : float, 
        dim : int,
        max_num_approx : int,
        adaptive : bool = True
    ) -> tuple:
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
    working_simplex = init_simplex
    max_diameter = init_simplex.calc_diameter()
    simplexes = []
    output_simplexes = []
    simplexes.append(copy.deepcopy(working_simplex))
    output_simplexes.append(copy.deepcopy(working_simplex))

    #проверки симплекса на наличие в нем корня
    if init_simplex.calc_topological_degree(func) == 0:
        #адаптивная стратегия поиска корня
        if adaptive:
            lines_info = __check_level_lines(func, working_simplex)
            if not lines_info[0]:
                x0 = np.sum(working_simplex.points, axis=0) / working_simplex.order
                sgn_matrix = Simplex.calc_sgn_matrix(func, working_simplex)
                column_sgn = np.all(sgn_matrix == sgn_matrix[0,:], axis = 0)
                try:
                    sgn_id = int(np.where(column_sgn == True)[0][0])
                except Exception:
                    sgn_id = 0

                sgn = int(sgn_matrix[0][sgn_id])

                def sgn_fn(f : function, sgn : int, sgn_id : int) -> function:
                    def g(x : np.ndarray):
                        return sgn * f(x)[sgn_id]
                    return g
                
                res = minimize(sgn_fn(func, sgn, sgn_id), x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
                
                if res.success and (func(res.x) < epsilon2).all() and working_simplex.check_point(res.x):
                    return {
                        "solution_steps" : output_simplexes, 
                        "solution" : res.x, 
                        "success" : True,  
                        "message" : "Корень найден"}
                
                return {
                    "solution_steps" : output_simplexes, 
                    "solution" : None, 
                    "success" : False, 
                    "message" : "Нет гарантии существования корня"
                    }
            else:
                working_simplex = lines_info[1]
        else:
            return {
                "solution_steps" : output_simplexes, 
                "solution" : None, 
                "success" : False, 
                "message" : "Нет гарантии существования корня"
                }
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
            return {
                "solution_steps" : output_simplexes, 
                "solution" : approx_solution, 
                "success" : True, 
                "message" : "Корень найден"
            }
        
        if (working_simplex.calc_point_norms(func) < epsilon2).any():
            approx_solution = np.array(working_simplex.points)[
                (working_simplex.calc_point_norms(func) < epsilon2)
            ][0]
            return {
                "solution_steps" : output_simplexes, 
                "solution" : approx_solution, 
                "success" : True, 
                "message" : "Корень найден"
            }
        
        if m == max_num_approx:
            working_simplex, m = __top_deg_stage(
                working_simplex, simplexes, func, 
                max_diameter, epsilon1, m, max_num_approx, dim
            )
            max_diameter = working_simplex.calc_diameter()
            output_simplexes.append(working_simplex)  
            continue

        #стадия обычной проверки симплекса
        simplex1, simplex2, mid_point = working_simplex.bisect(return_mid=True)
        if (func(mid_point) < epsilon1).any():
            simplex1, simplex2 = working_simplex.bisect(scale=0.3)
        in_simplex1 = simplex1.transform(func).check_point(np.zeros(dim))
        in_simplex2 = simplex2.transform(func).check_point(np.zeros(dim))

        if in_simplex1:
            m += 2
            simplexes.append(simplex1)
            simplexes.append(simplex2)
            working_simplex = simplex1
            max_diameter = working_simplex.calc_diameter()
            output_simplexes.append(working_simplex)   
            continue

        elif in_simplex2:
            m += 2
            simplexes.append(simplex1)
            simplexes.append(simplex2)
            working_simplex = simplex2
            max_diameter = working_simplex.calc_diameter()
            output_simplexes.append(working_simplex)   
            continue

        #стадия проверки отраженного симплекса
        simplex1, simplex2, flag = working_simplex.reflection(func)
        in_simplex1 = simplex1.transform(func).check_point(np.zeros(dim))
        in_simplex2 = simplex2.transform(func).check_point(np.zeros(dim))
        
        if in_simplex1 and flag:
            m += 2
            simplexes.append(simplex1)
            simplexes.append(simplex2)
            working_simplex = simplex1

        elif in_simplex2 and flag:
            m += 2
            simplexes.append(simplex1)
            simplexes.append(simplex2)
            working_simplex = simplex2

        else:
            working_simplex, m = __top_deg_stage(
                working_simplex, simplexes, func, 
                max_diameter, epsilon1, m, max_num_approx, dim
            )
        
        if working_simplex is None:
            return {
                "solution_steps" : output_simplexes, 
                "solution" : None, 
                "success" : False, 
                "message" : "Корень не найден"
            }

        max_diameter = working_simplex.calc_diameter()
        output_simplexes.append(working_simplex)   


def __char_polyh_bisection(
    func    : function,
    epsilon : float,
    delta   : float,
    dim     : int,
    init_vertice : np.ndarray,
    step_sizes   : np.ndarray,
    max_iter : int = 1000
) -> tuple:
    """
    Purpose: Perform characteristic polyhedron bisection until diameter is below epsilon
    
    :param func: Target function for bisection
    :param epsilon: Maximum allowed diameter threshold
    :param delta: Delta value for polyhedron construction
    :param dim: Dimension of the space
    
    :return: tuple: (list of intermediate polyhedrons, final solution)
    """
    polyh = CharPolyhedron(
        dim=dim, 
        func=func, 
        delta=delta,
        init_vertice=init_vertice,
        step_sizes=step_sizes
    )
    output_polyhs = []
    solution = None

    i = 0

    while (polyh.calc_diameter() > epsilon and i < max_iter):
        polyh.bisect(func)
        output_polyhs.append(copy.deepcopy(polyh))

        for vertice in polyh.vertices:
            if (np.abs(func(vertice)) < epsilon).all():
               solution = vertice
               break 
        
        if solution is not None:
            break

        i += 1

    if solution is None:
        solution = polyh.calc_solution()
    
    return output_polyhs, solution


def solve_det_system(
        func : function, 
        method_name : str,
        **kwargs
    ) -> dict:
    """
    Purpose: Solve deterministic system using specified method
    
    :param func: Target function to solve
    :param method_name: Name of solution method ('topdeg' or 'charpolyh')
    :param kwargs: Method-specific parameters
    
    :return: tuple: Solution results (format depends on method)
    
    :raises KeyError: If required kwargs are missing for the specified method
    """
    if method_name == "topdeg":
        init_simplex   = kwargs.get('init_simplex')
        epsilon1       = kwargs.get('epsilon1')
        epsilon2       = kwargs.get('epsilon2')
        dim            = kwargs.get('dim')
        max_num_approx = kwargs.get('max_num_approx')

        if (init_simplex is None or
            epsilon1 is None or
            epsilon2 is None or
            dim is None or
            max_num_approx is None):
            raise KeyError("Incorrect kwargs for topdeg method")
        
        res =  __topologic_deg_bisect(
            func, 
            init_simplex,
            epsilon1, 
            epsilon2, 
            dim, 
            max_num_approx
        )
        if (res['solution'] is not None) and (func(res['solution']) < max(epsilon1, epsilon2)).all():
            return res
    
        res['solution'] = None
        res['success'] = False
        res['message'] = "Корень не найден"
        return res
    
    elif method_name == "charpolyh":
        epsilon   = kwargs.get('epsilon')
        dim       = kwargs.get('dim')
        delta     = kwargs.get('delta')
        max_iter  = kwargs.get('max_iter')
        init_vertice = kwargs.get('init_vertice')
        step_sizes   = kwargs.get('step_sizes')
        
        if (epsilon      is None or
            dim          is None or
            max_iter     is None or
            delta        is None or
            init_vertice is None or
            step_sizes   is None):
            raise KeyError("Incorrect kwargs for charpolyh method")
        
        return  __char_polyh_bisection(
            func, 
            epsilon,
            delta,
            dim,
            init_vertice,
            step_sizes,
            max_iter
        )


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
    u_k = np.zeros((dim + 1, dim + 1))
    for k in range(dim + 1):
        u_k[k, k] = 1.0
    u_k[-1, :] = (-1.0)/(1 + np.sqrt(dim + 1)) * np.ones(dim + 1)
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
    
    apex = c.coords + (1.0 / (M *  (dim + 1))) * sum_terms
    apex[dim] = (sum_f/(dim + 1)) - M * (dim + 1)
    apex = apex
    height = M * dim * r - (1.0/(dim + 1)) * (sum_f - (dim + 1) * m) 
    
    base_vertices = []
    for k in range(dim + 1):
        vertex = apex.coords.copy()
        vertex[:dim] += (height / M) * u_k[k][:dim]
        vertex[dim] += height
        base_vertices.append(vertex)
    
    return StandartSimplex(base_vertices + [apex], dim + 1, dim + 1)
