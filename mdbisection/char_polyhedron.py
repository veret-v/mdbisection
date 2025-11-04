from __future__ import annotations

import copy
import itertools as it
from math import factorial

import numpy as np


class CharPolyhedron:
    """
    Purpose: Represents a geometric characteristic polyhedron with 
             vertices in n-dimensional space
             Provides operations like construction, bisection
    """
    _vertices       : np.ndarray
    _dim            : int
    _sgn_ref_matrix : np.ndarray

    def __init__(
            self, 
            dim : int, 
            func : function, 
            delta: float, 
            init_vertice : np.ndarray, 
            step_sizes : np.ndarray
        ) -> None:
        """
        Purpose: Initialize characteristic polyhedron with given dimension,
                 function and delta value
        
        :param dim: Dimension of the space
        :param func: Function used for construction
        :param delta: Delta value for construction
        
        :return: None
        """
        self._dim            = dim
        self._sgn_ref_matrix = self.__bin_permut(dim)
        self._vertices       = self.__construct(func, delta, init_vertice, step_sizes)
        
    @property
    def dim(self):
        """
        Purpose: Get space dimension
        
        :param_data: None (getter) 
        
        :return: int: Current dimension
        """
        return self._dim

    @property
    def vertices(self):
        """
        Purpose: Get characteristic polyhedron vertices
                 Maintains deep copies when setting
        
        :param_data: None (getter) 
        
        :return: list[vertice]: Current vertices
        """
        return self._vertices
    
    def calc_diameter(self):
        """
        Purpose: Calculate diameter of the polyhedron
        
        :param_data: None 
        
        :return: float: Diameter of the polyhedron
        """
        diam = 0

        for i in range(2 ** self.dim):
            for j in range(self.dim):
                k = int(i - 2 ** (self.dim - j - 1) * self._sgn_ref_matrix[i, j])
                length = ((self.vertices[i] - self.vertices[k]) ** 2).sum()

                if length > diam:
                    diam = length
        
        return diam

    def calc_solution(self):
        diag = 0
        sol  = self.vertices[0]

        for k in range(2 ** self.dim):
            l = int(2 ** self.dim - 1 - k)
            length = ((self.vertices[l] - self.vertices[k]) ** 2).sum()

            if length > diag:
                diag = length
                sol = (self.vertices[l] + self.vertices[k]) / 2

        return sol

    def bisect(self, func : function):
        """
        Purpose: Perform bisection operation on the polyhedron
        
        :param func: Function used for bisection
        
        :return: None
        """
        for i in range(2 ** self.dim):
            for j in range(self.dim):
                k = int(i - 2 ** (self.dim - j - 1) * self._sgn_ref_matrix[i, j])
                mid_point = (self.vertices[i] + self.vertices[k]) / 2
        
                if ((func(self.vertices[i]) >= 0) == (func(mid_point) >= 0)).all():
                    self.vertices[i] = mid_point.copy()

                elif ((func(self.vertices[k]) >= 0) == (func(mid_point) >= 0)).all():
                    self.vertices[k] = mid_point.copy()

                else:
                    self.vertices[i], self.vertices[k] = self.__relaxation(mid_point, i, k, func)

    def __relaxation(self, mid_point : np.ndarray, i : int, k : int, func : function) -> tuple:
        """
        Purpose: Perform relaxation operation during bisection
        
        :param mid_point: Mid point between vertices
        :param i: Index of first vertex
        :param k: Index of second vertex
        :param func: Function used for relaxation
        
        :return: tuple: New vertices after relaxation
        """
        new_vertice = mid_point

        for _ in range(2):
            for vertice in self.vertices:

                if ((func(vertice) >= 0) == (func(new_vertice) >= 0)).all():
                    new_vertice = new_vertice * 2 - vertice
                    break
            
            if ((func(self.vertices[i]) >= 0) == (func(new_vertice) >= 0)).all():
                return new_vertice, self.vertices[k]

            elif ((func(self.vertices[k]) >= 0) == (func(new_vertice) >= 0)).all():
                return self.vertices[i], new_vertice
        
        return self.vertices[i], self.vertices[k]
            

    def __construct(
            self, 
            func : function, 
            delta : float, 
            init_vertice : np.ndarray, 
            step_sizes : np.ndarray
        ) -> np.ndarray:
        """
        Purpose: Construct initial vertices of the polyhedron
        
        :param func: Function used for construction
        :param delta: Delta value for construction
        
        :return: np.ndarray: Initial vertices
        """
        bin_marix = (self._sgn_ref_matrix == 1)

        vertices  = init_vertice.copy()
        for i in range(2 ** self.dim - 1):
            vertices = np.vstack([vertices, init_vertice])

        stepsizes = np.diag(step_sizes)
        init_vertices  = vertices + bin_marix.dot(stepsizes)
        sgn_matrix = np.stack([func(vertice) >= 0 for vertice in init_vertices])

        result_vertices = np.empty((2 ** self.dim, self.dim))
        correct_vertices = np.ones(2 ** self.dim)

        for i in range(2 ** self.dim):
            for j in range(2 ** self.dim):
                if (bin_marix[i] == sgn_matrix[j]).all() and correct_vertices[i]:
                    result_vertices[i] = init_vertices[j].copy()
                    correct_vertices[i] = 0
                    break

        for i in range(2 ** self.dim):
            for j in range(self.dim):
                k = int(i - 2 ** (self.dim - j - 1) * self._sgn_ref_matrix[i, j])
                s = np.where(np.abs(init_vertices[k] - init_vertices[i]) < 1e-8)[0][0]

                alpha = min(abs(init_vertices[k, s]), abs(init_vertices[i, s]))
                beta  = abs(init_vertices[k, s] - init_vertices[i, s])
                r_s = self.__solve_eq(func, init_vertices[k], s, delta, alpha, alpha + beta)
                dstar = delta + np.random.rand() * (min(r_s - alpha, alpha + beta - r_s) - delta)
 
                new_vertice_1 = init_vertices[i].copy()
                new_vertice_2 = init_vertices[i].copy()
                new_vertice_1[s] = r_s + dstar
                new_vertice_2[s] = r_s - dstar

                for p in range(2 ** self.dim):

                    if ((func(new_vertice_1) >= 0) == bin_marix[p]).all() and correct_vertices[p]:
                        result_vertices[p] = new_vertice_1
                        correct_vertices[p] = 0
                        break

                    elif ((func(new_vertice_2) >= 0) == bin_marix[p]).all() and correct_vertices[p]:
                        result_vertices[p] = new_vertice_2
                        correct_vertices[p] = 0
                        break
                
        return result_vertices
               
               
    @staticmethod
    def __bin_permut(n : int) -> np.ndarray:
        """
        Purpose: Generate binary permutation matrix
        
        :param n: Dimension for permutation
        
        :return: np.ndarray: Binary permutation matrix
        """
        bin_perm = np.empty((2 ** n, n))
        for i, perm in enumerate(it.product([-1, 1], repeat=n)):
            bin_perm[i, :] = perm
        return bin_perm
    
    @staticmethod
    def __solve_eq(
        func  : function, 
        val   : np.ndarray, 
        s     : int, 
        delta : float,
        left  : float,
        right : float
    ) -> float:
        """
        Purpose: Solve equation using bisection method
        
        :param func: Function to solve
        :param val: Initial value
        :param s: Index parameter
        :param delta: Delta value
        :param left: Left boundary
        :param right: Right boundary
        
        :return: float: Solution within given boundaries
        """
        middle = (right + left) / 2

        while right - left > delta:
            mid_val      = val.copy()
            left_val     = val.copy()
            right_val    = val.copy()
            mid_val[s]   = middle
            left_val[s]  = left
            right_val[s] = right

            if func(mid_val)[s] * func(left_val)[s] < 0:
                right = middle
            else:
                left = middle
            
            middle = (right + left) / 2
    
        return middle


        