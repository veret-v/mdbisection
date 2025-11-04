from __future__ import annotations

import math
import itertools
import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from numpy import heaviside
from matplotlib.patches import Polygon
from scipy.optimize import shgo
from scipy.optimize import root

from mdbisection.simplex import Simplex
from mdbisection.simplex_system import StandartSimplex
from mdbisection.algorithms import solve_det_system, optimize
from mdbisection.visualize import draw_algorithm_2d, draw_algorithm_3d, draw_optimize_3d
from mdbisection.visualize import test_draw_3d, test_draw_2d
 


def F2_point_level(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = X[0] ** 2 + X[1] ** 2
    Y[1] = X[0] ** 2 - X[1] ** 2
    return Y

def F2_closed_level(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = X[0] ** 2 + X[1] - 0.1
    Y[1] = X[0] ** 2 - X[1] ** 2
    return Y

def F2_H(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    if X[0] >= 0:
        Y[0] = X[1] - X[0] ** 2
    else:
        Y[0] = X[1] - X[0] - 1
    Y[1] = X[1] - 2 * X[0] ** 2 + 1
    return Y


def F2_deriv(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = X[1] - abs(X[0]) - X[0] ** 2
    Y[1] = X[1] - 2 * X[0] - 1
    return Y


def F2_1(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = (X[0] - 0.1) ** 2 + X[1] - 0.1
    Y[1] = (X[1] - 0.1) ** 2 + X[0] - 0.1
    return Y


def F2_2(X : np.ndarray) -> np.ndarray:
    Y = np.zeros(2)
    Y[0] = X[0] + 0.5 * (X[0] - X[1]) ** 3 - 1.0
    Y[1] = 0.5 * (X[1] - X[0]) ** 3 + X[1]
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


def F2_draw_deriv(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = Y - np.abs(X) - X ** 2
    Z2 = Y - 2 * X - 1
    return Z1, Z2

def F2_draw_H(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = Y - heaviside(X, 0) * X ** 2 - (1 - heaviside(X, 0) * (X + 1))
    Z2 = Y - 2 * X ** 2 + 1
    return Z1, Z2

def F_draw_2D_point_level(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = X ** 2 + Y ** 2
    Z2 = X ** 2 - Y ** 2
    return Z1, Z2

def F_draw_2D_closed_level(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = X ** 2 + Y ** 2 - 0.1
    Z2 = X ** 2 - Y ** 2
    return Z1, Z2


def F_draw_2D_1(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = (X - 0.1) ** 2 + Y - 0.1
    Z2 = (Y - 0.1) ** 2 + X - 0.1
    return Z1, Z2

def F_draw_2D_2(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = X + 0.5 * (X - Y) ** 3 - 1.0
    Z2 = 0.5 * (Y - X) ** 3 + Y
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

def F2_optim1(X : np.ndarray) -> float:
    return max(3 ** 0.5 * X[0] + X[1], -2 * X[1], X[1] - 3 ** 0.5 * X[0])

def F2_optim1_draw(X : np.ndarray, Y : np.ndarray) -> float:
    return np.max(3 ** 0.5 * X + Y, -2 * Y, Y - 3 ** 0.5 * X)
   
    
if __name__ == '__main__':

    start_time = time.time()
    init_simplex_3 = Simplex(
            np.array([
                [1, -1], 
                [2, 2],
                [-3, 2]   
            ]), 2, 2)    
    init_simplex_4 = Simplex(
            np.array([
                [0.22, 0.2],
                [0.3, 0.26],
                [0.15, 0.25]
            ]), 2, 2)    
    init_simplex_2 = Simplex(
            np.array([
                [0.34111, 0.34222], 
                [-0.32333, 0.0111], 
                [0.02222, -0.35666]
            ]), 2, 2)      
    init_simplex_1 = Simplex(
            np.array([
                [0.33, 0.33], 
                [-0.33, 0], 
                [0.0, -0.33]
            ]), 2, 2)        
    res = solve_det_system(F2_H, method_name='topdeg', init_simplex=init_simplex_3, epsilon1=1e-9, epsilon2=1e-8, dim=2, max_num_approx=5)
    print(res['solution'])
    draw_algorithm_2d(res['solution_steps'], F2_draw_H) 
    sol = root(F2_H, [1, 1], method='lm')
    print(sol.x)
    print(F2_H(sol.x))
    if res['solution'] is not None:
        print(F2_H(res['solution']))
    print("--- %s seconds ---" % (time.time() - start_time))


    
    # init_simplex_3 = Simplex(
    #         np.array([
    #             [-1, -1, -1], 
    #             [1, 0, 0], 
    #             [0, 1, 0],
    #             [0, 0, 1],
    #         ]), 3, 3)       
    # res = solve_det_system(F3, method_name='topdeg', init_simplex=init_simplex_3, epsilon1=1e-8, epsilon2=1e-6, dim=3, max_num_approx=10)
    # print(res['solution'])
    # draw_algorithm_3d(res['solution_steps'], F_draw_3D) 

    # init_simplex_5 = Simplex(
    #         np.array([
    #             [0.2, 0.2, 0.2, 0.2, 0.2], 
    #             [-0.2, 0, 0, 0, 0], 
    #             [0, -0.2, 0, 0, 0], 
    #             [0, 0, -0.2, 0, 0], 
    #             [0, 0, 0, -0.2, 0], 
    #             [0, 0, 0, 0, -0.2], 
    #         ]), 5, 6)
    # res = solve_det_system(F5, init_simplex_5, epsilon1=1e-8, epsilon2=1e-6, dim=5, max_num_approx=30)
    # print(res['solution'])

