from __future__ import annotations

import math
import itertools
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mdbisection.simplex import Simplex, Point
from mdbisection.simplex_system import StandartSimplex
from mdbisection.algorithms import solve_det_system, optimize
from mdbisection.visualize import draw_algorithm_2d, draw_algorithm_3d, draw_optimize_3d
from mdbisection.visualize import test_draw_3d, test_draw_2d


def F2(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = X[0] ** 2 - 4 * X[1]
    Y[1] = X[1] ** 2 - 2 * X[0] + 4 * X[1]
    return Y


def F3(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(3)
    Y[0] = (X[0] - 0.1) ** 2 + X[1] - 0.1
    Y[1] = (X[1] - 0.1) ** 2 + X[2] - 0.1
    Y[2] = (X[2] - 0.1) ** 2 + X[0] - 0.1
    return Y


def F5(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(5)
    Y[0] = (X[0] - 0.1) ** 2 + X[1] - 0.1
    Y[1] = (X[1] - 0.1) ** 2 + X[2] - 0.1
    Y[2] = (X[2] - 0.1) ** 2 + X[3] - 0.1
    Y[3] = (X[3] - 0.1) ** 2 + X[4] - 0.1
    Y[4] = (X[4] - 0.1) ** 2 + X[0] - 0.1
    return Y


def F_draw_2D(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = X ** 2 - 4 * Y
    Z2 = Y ** 2 - 2 * X + 4 * Y
    return Z1, Z2


def F_draw_3D(X : np.ndarray, Y : np.ndarray, Z : np.ndarray) -> tuple:
    Z1 = (X - 0.1) ** 2 + Y - 0.1
    Z2 = (Y - 0.1) ** 2 + Z - 0.1
    Z3 = (Z - 0.1) ** 2 + X - 0.1
    return Z1, Z2, Z3

def F2_optim(X : np.ndarray) -> float:
    return -math.exp(-X[0] ** 2) * math.sin(X[0]) + abs(X[1])

def F2_optim_draw(X : np.ndarray, Y : np.ndarray) -> float:
    return -np.exp(-X ** 2) * np.sin(X) + np.abs(Y)
   
    
if __name__ == '__main__':

    # init_simplex_opt = StandartSimplex(
    #         [
    #             Point(np.array([0, 0, -3]), 3), 
    #             Point(np.array([2, 0, 1]), 3), 
    #             Point(np.array([0, 1, 1]), 3),
    #             Point(np.array([0, -1, 1]), 3)
    #         ], 3, 4)         
    # simplexes, optimum = optimize(1, F2_optim, 2, Point(np.array([0, 0, 0]), 3), 20, 1e-1, init_simplex_opt)
    # draw_optimize_3d(simplexes[:15], F2_optim_draw)
    # draw_optimize_3d(simplexes[:10], F2_optim_draw)
    # draw_optimize_3d(simplexes[:5], F2_optim_draw)

    init_simplex_2 = Simplex(
            [
                Point(np.array([0.33, 0.33]), 2), 
                Point(np.array([-0.33, 0]), 2), 
                Point(np.array([0, -0.33]), 2)
            ], 2, 2)       
    simplexes2, solution = solve_det_system(F2, init_simplex_2, epsilon1=1e-8, epsilon2=1e-6, dim=2, max_num_approx=100)
    print(solution)
    draw_algorithm_2d(simplexes2, F_draw_2D) 

    # init_simplex_3 = Simplex(
    #         [
    #             Point(np.array([-1, -1, -1]), 3), 
    #             Point(np.array([1, 0, 0]), 3), 
    #             Point(np.array([0, 1, 0]), 3),
    #             Point(np.array([0, 0, 1]), 3)
    #         ], 3, 3)       
    # simplexes3, solution = solve_det_system(F3, init_simplex_3, epsilon1=1e-8, epsilon2=1e-6, dim=3, max_num_approx=10)
    # print(solution)
    # draw_algorithm_3d(simplexes3, F_draw_3D) 

    # init_simplex_5 = Simplex(
    #         [
    #             Point(np.array([0.2, 0.2, 0.2, 0.2, 0.2]), 5), 
    #             Point(np.array([-0.2, 0, 0, 0, 0]), 5), 
    #             Point(np.array([0, -0.2, 0, 0, 0]), 5), 
    #             Point(np.array([0, 0, -0.2, 0, 0]), 5), 
    #             Point(np.array([0, 0, 0, -0.2, 0]), 5), 
    #             Point(np.array([0, 0, 0, 0, -0.2]), 5), 
    #         ], 5, 6)
    # simplexes5, solution = solve_det_system(F5, init_simplex_5, epsilon1=1e-8, epsilon2=1e-6, dim=5, max_num_approx=30)
    # print(solution)