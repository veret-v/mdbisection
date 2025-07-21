from __future__ import annotations

import math
import itertools

import numpy as np
import copy
from collections import Counter

from .simplex import Simplex


class StandartSimplex(Simplex):
    """
    Purpose: Represents a standard simplex with base points, apex, and height properties
             Provides methods for simplex reduction and elimination operations
    """

    _base   : np.ndarray
    _apex   : np.ndarray
    _height : float

    def __init__(self, points : np.ndarray, dim : int, order : int) -> None:
        """
        Purpose: Represents a standard simplex with base points, apex, and height properties
                Provides methods for simplex reduction and elimination operations
        
        :param_data: points: List of Point objects defining the simplex vertices
        :param_data: dim: Dimension of the simplex
        :param_data: order: Order of the simplex
        
        :return: StandartSimplex instance
        """
        self._base, self._apex, self._height = self.__check_calc_standart(points, dim, order)
        super().__init__(points, dim, order) 

    @property
    def apex(self):
        """
        Purpose: Get the apex point of the standard simplex
                 The apex is the vertex with unique z-coordinate in the base-apex pair
        
        :param_data: None
        
        :return: Point: The apex point of the simplex
        """
        return self._apex
    
    @property
    def base(self):
        """
        Purpose: Get the base points of the standard simplex
                 The base forms the (dim-1)-dimensional face opposite the apex
        
        :param_data: None
        
        :return: list[Point]: List of base points
        """
        return self._base

    @property
    def height(self):
        """
        Purpose: Get the height of the standard simplex
                 The perpendicular distance between apex and base plane
        
        :param_data: None
        
        :return: float: Height value
        """
        return self._height
    
    def reduction(self, func_val : float) -> list[StandartSimplex]:
        """
        Purpose: Perform reduction operation on the simplex based on function value
                 Generates new simplices by reducing height according to function value
        
        :param_data: func_val: Function value at simplex apex coordinates
        
        :return: list[StandartSimplex] or None: List of reduced simplices or None if no reduction possible
        """
        gen_simplexes = []
        support_vecs = []
        new_base = []
        h_new = 0

        if self.height <=  func_val - self.apex[self.dim- 1] <= self.height * (self.dim + 1):
            h_new = self.height - 1 / (self.dim + 1) * (func_val - self.apex[self.dim- 1])

            for base_point in self.base:
                support_vec = (self.apex - base_point) 
                support_vecs.append(support_vec)

            for support_vec1 in support_vecs:
                new_apex = self.apex - support_vec1 * (func_val - self.apex[self.dim- 1]) / (self.height * (self.dim + 1))
                new_std_simplex = StandartSimplex(
                    np.stack([new_apex] + [new_apex - support_vec2 * h_new / self.height for support_vec2 in support_vecs]), 
                    self.dim, self.order
                )
                gen_simplexes.append(new_std_simplex)

        elif 0 <= func_val - self.apex[self.dim- 1] < self.height:
            h_new = self.dim / (self.dim + 1) * (func_val - self.apex[self.dim- 1])

            for base_point in self.base:
                support_vec = (self.apex - base_point) 
                support_vecs.append(support_vec)

            for support_vec1 in support_vecs:
                new_apex = self.apex - support_vec1 * (func_val - self.apex[self.dim- 1] - h_new) / self.height
                new_std_simplex = StandartSimplex(
                    np.stack([new_apex] + [new_apex - support_vec2 * h_new / self.height for support_vec2 in support_vecs]), 
                    self.dim, self.order
                )
                gen_simplexes.append(new_std_simplex)

        else:
            return None
    
        return gen_simplexes

    def elimination(self, el_height):
        """
        Purpose: Eliminate parts of simplex below specified height threshold
                 Returns None if simplex is completely below threshold
        
        :param_data: el_height: Elimination height value in last coordinate
        
        :return: StandartSimplex or None: Modified simplex or None if eliminated
        """
        if self.apex[self.dim - 1] >= el_height:
            return None
        
        if self.apex[self.dim - 1] + self.height <= el_height:
            return self
        
        new_base  = []

        for base_point in self.base:
            new_base_point = self.apex - (self.apex - base_point) * (el_height - self.apex[self.dim- 1]) / self.height
            new_base.append(new_base_point)

        try:
            return StandartSimplex(np.stack(new_base + [self.apex]), self.dim, self.order)
        except TypeError:
            return None

    @staticmethod
    def __check_calc_standart(points : np.ndarray, dim : int, order : int):
        mask = np.array([point[dim - 1] for point in points])
        unique_elem = list(set(mask))
        counted_elem = Counter(mask)

        if len(unique_elem) != 2:
            raise TypeError("Not standart simplex : not only 2 heights")
                
        if counted_elem[unique_elem[0]] == 1:
            min_point = points[mask == unique_elem[0]].reshape(-1)
            base      = np.delete(points, np.where(mask == unique_elem[0])[0], axis=0)
            height    = (mask[mask == unique_elem[1]] - mask[mask == unique_elem[0]])[dim - 1]
    
        elif counted_elem[unique_elem[1]] == 1:
            
            min_point = points[mask == unique_elem[1]].reshape(-1)
            base      = np.delete(points, np.where(mask == unique_elem[1])[0], axis=0)
            height    = (mask[mask == unique_elem[0]] - mask[mask == unique_elem[1]])[dim - 1] 
            
        if height < 0:
                raise TypeError(f"Not standart simplex: height({height}) < 0")
        
        return base, min_point, height
    
class SimplexSystem:
    """
    Purpose: Manages a system of standard simplices with collective operations
            Provides reduction and elimination for entire simplex systems
    """
    _simplexes  : list[StandartSimplex]
    _dim        : int
    _order      : int
    _lowest_top : np.ndarray
    _variation  : float
    
    def __init__(self, simplexes, dim, order, lowest_top):
        """
        Purpose: Manages a system of standard simplices with collective operations
                Provides reduction and elimination for entire simplex systems
        
        :param_data: simplexes: List of StandartSimplex objects
        :param_data: dim: System dimension
        :param_data: order: System order
        :param_data: lowest_top: Point with minimal function value
        
        :return: SimplexSystem instance
        """
        self._simplexes = simplexes
        self._dim = dim
        self._order = order
        self._lowest_top = lowest_top
        self._variation = self.__calc_variation()

    @property
    def simplexes(self):
        """
        Purpose: Get or set the list of simplices in the system
                 Validates that all simplices have matching dimensions
        
        :param_data: None (getter) or simplexes (setter)
        
        :return: list[StandartSimplex]: Current simplex collection
        """
        return self._simplexes
    
    @simplexes.setter
    def simplexes(self, simplexes):
        """
        Purpose: Get or set the list of simplices in the system
                 Validates that all simplices have matching dimensions
        
        :param_data: None (getter) or simplexes (setter)
        
        :return: list[StandartSimplex]: Current simplex collection
        """
        self._simplexes = copy.deepcopy(simplexes)

    @property
    def dim(self):
        """
        Purpose: Get or set the system dimension
                 Validates dimension consistency across all simplices
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current system dimension
        """
        return self._dim
    
    @dim.setter
    def dim(self, dim):
        """
        Purpose: Get or set the system dimension
                 Validates dimension consistency across all simplices
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current system dimension
        """
        if any(dim != simplex.dim for simplex in self.simplexes):
            raise ValueError(f"Pазмерность симплексов не совпадает с размерностью({dim})")
        self._dim = dim

    @property
    def order(self):
        """
        Purpose: Get or set the system order
                 Validates order consistency across all simplices
        
        :param_data: None (getter) or order (setter)
        
        :return: int: Current system order
        """
        return self._order
    
    @order.setter
    def order(self, order):
        """
        Purpose: Get or set the system order
                 Validates order consistency across all simplices
        
        :param_data: None (getter) or order (setter)
        
        :return: int: Current system order
        """
        if any(order != simplex.order for simplex in self.simplexes):
            raise ValueError(f"Порядок симплексов не совпадает с размерностью({order})")
        self._order = order

    @property
    def lowest_top(self):
        """
        Purpose: Get the point with minimal function value in the system
        
        :param_data: None
        
        :return: Point: The lowest top point
        """
        return self._lowest_top
    
    @property
    def variation(self):
        """
        Purpose: Get the variation measure of the simplex system
                 Calculated as max height minus min height across all simplices
        
        :param_data: None
        
        :return: float: Variation value
        """
        return self._variation
    
    def reduction(self, func : function) -> SimplexSystem:
        """
        Purpose: Perform reduction operation on all simplices using given function
                 Creates new simplex system with reduced simplices
        
        :param_data: func: Function to evaluate at simplex apex coordinates
        
        :return: SimplexSystem: New system with reduced simplices
        """
        reducted_simplexes = []
        lowest_top_red = self.lowest_top.copy()
        for std_simplex in self.simplexes:
            reducted_simplex = std_simplex.reduction(func(std_simplex.apex[:self.dim - 1]))

            if reducted_simplex is not None:
                reducted_simplexes.append(reducted_simplex)

            if func(std_simplex.apex[:self.dim - 1]) < lowest_top_red[self.dim - 1]:
                lowest_top_red = std_simplex.apex.copy()
                lowest_top_red[-1] = func(std_simplex.apex[:self.dim - 1])

        return SimplexSystem(list(
            itertools.chain(
                *reducted_simplexes
            )), self.dim, self.order, lowest_top_red)
    
    def elimination(self):
        """
        Purpose: Eliminate parts of simplices below current lowest_top height
                 Returns new system with only surviving simplex portions
        
        :param_data: None
        
        :return: SimplexSystem: New system after elimination
        """
        eliminated_simplexes = []
        for std_simplex in self.simplexes:
            eliminated_simplex = std_simplex.elimination(self.lowest_top[self.dim - 1])

            if eliminated_simplex is not None:
                eliminated_simplexes.append(eliminated_simplex)

        return SimplexSystem(eliminated_simplexes, self.dim, self.order, self.lowest_top)
    
    def __calc_variation(self):
        """
        Purpose: Calcs variation of system by max(x_i + h_i) - min(x_i)
        
        :param_data: None
        
        :return: variation: float value
        """
        min_points = np.array([std_simplex.apex[self.dim - 1] for std_simplex in self.simplexes])
        max_heights = np.array([std_simplex.apex[self.dim - 1] + std_simplex.height for std_simplex in self.simplexes])
        return np.max(max_heights) - np.min(min_points)
    
