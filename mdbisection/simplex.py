from __future__ import annotations

import copy
import itertools

import numpy as np


class Simplex:
    """
    Purpose: Represents a geometric simplex with vertices in n-dimensional space
            Provides operations like reflection, bisection, and topological degree calculation
    """
    _points : np.ndarray
    _dim    : int
    _order  : int

    def __init__(self, points : np.ndarray, dim : int, order : int) -> None:
        """
        Purpose: Represents a geometric simplex with vertices in n-dimensional space
                Provides operations like reflection, bisection, and topological degree calculation
        
        :param_data: points: List of vertex Points
        :param_data: dim: Space dimension
        :param_data: order: Simplex order (number of vertices - 1)
        
        :return: Simplex instance
        """
        self._dim = dim
        self._order = order
        self._points = points

    @property
    def points(self):
        """
        Purpose: Get or set simplex vertices
                 Maintains deep copies when setting
        
        :param_data: None (getter) or points (setter)
        
        :return: list[Point]: Current vertices
        """
        return self._points
    
    @points.setter
    def points(self, points : np.ndarray):
        """
        Purpose: Get or set simplex vertices
                 Maintains deep copies when setting
        
        :param_data: None (getter) or points (setter)
        
        :return: list[Point]: Current vertices
        """
        if points.shape[0] != self.order or points.shape[1] != self.dim:
            raise TypeError("Not simplex : incorrect size of points matrix")
        self._points = points.copy()

    @property
    def dim(self):
        """
        Purpose: Get or set space dimension
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current dimension
        """
        return self._dim
    
    @dim.setter
    def dim(self, dim : int):
        """
        Purpose: Get or set space dimension
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current dimension
        """
        self._dim = dim
    
    @property
    def order(self):
        """
        Purpose: Get or set simplex order
        
        :param_data: None (getter) or order (setter)
        
        :return: int: Current order
        """
        return self._order
    
    @order.setter
    def order(self, order : int):
        """
        Purpose: Get or set simplex order
        
        :param_data: None (getter) or order (setter)
        
        :return: int: Current order
        """
        self._order = order

    def calc_point_norms(self, func : function) -> np.ndarray:
        """
        Purpose: Calculate norms of radius vectors after function transformation
        
        :param_data: func: Transformation function Rⁿ → Rⁿ
        
        :return: np.ndarray: Array of norm values
        """
        return np.array([np.sum(func(point)**2) ** 0.5 for point  in  self.points])
    
    def reflection(self,  func : function) -> tuple:
        """
        Purpose: Generate new simplices through reflection operation
                 Based on function values at vertices
        
        :param_data: func: Function used for reflection calculation
        
        :return: tuple: (simplex1, simplex2, flag) - New simplices and inclusion flag
        """
        k, l = self.__calc_max_egde()
        middle_point = (self.points[k] + self.points[l]) / 2
        i0 = 0
        if self.order == 2:
            k = 0
            l = 1
            i = 2
        else:
            iterable_index = list(set(range(self.order + 1)) - set((k, l)))
            if self.transform(func).check_point(middle_point):
                min_dist = Simplex(
                    np.stack([self.points[p] for p in (set(range(self.order + 1)) - set([0]))]), 
                    self.dim,
                    self.order
                ).transform(func).distance(np.zeros(self.order))
                i0 = iterable_index[0]

                for i in iterable_index:
                    dist = Simplex(
                        np.stack([self.points[p] for p in (set(range(self.order + 1)) - set([i]))]), 
                        self.dim,
                        self.order
                    ).transform(func).distance(np.zeros(self.order))

                    if min_dist > dist:
                        min_dist = dist
                        i0 = i
            else:
                v = func(self.points[k]) - func(self.points[l])
                fd = func(middle_point) - (func(self.points[k]) + func(self.points[l])) / 2
                w  = func(self.points[i0]) - (func(self.points[k]) + func(self.points[l])) / 2
            
                projection = lambda x : x - v * self.inner_prod(x, v) / self.inner_prod(v, v) 

                i0 = iterable_index[0]
                max_metric = self.inner_prod(projection(w), (projection(fd))) / (self.norm(projection(w)) * self.norm(projection(fd)))

                for i  in iterable_index:
                    w  = func(self.points[i]) - (func(self.points[k]) + func(self.points[l])) / 2
                    metric = self.inner_prod(projection(w), (projection(fd))) / (self.norm(projection(w)) * self.norm(projection(fd)))
                    if metric > max_metric:
                        max_metric = metric
                        i0 = i

        reflected_point = self.points[k] + self.points[l] - self.points[i0]
        flag = self.check_point(reflected_point)

        new_points = self.points[:]
        new_points = np.delete(np.vstack([new_points, [middle_point, reflected_point]]), i0, axis=0)
        simplex1_points = np.delete(new_points, k, axis=0)
        simplex2_points = np.delete(new_points, l, axis=0)
        return (
            Simplex(simplex1_points, self.dim, self.order), 
            Simplex(simplex2_points, self.dim, self.order),
            flag
        )
    
    def distance(self, target_point : np.ndarray) -> float:
        """
        Purpose: Calculate minimal distance to target point
                 Uses metric min(||Xi - Y||) where Xi are vertices
        
        :param_data: target_point: Point to measure distance to
        
        :return: float: Minimal distance value
        """
        min_dist = self.norm(self.points[0] - target_point)
        for point in self.points:
            dist = self.norm(point - target_point)
            if min_dist > dist:
                min_dist = dist
        return min_dist
    

    def transform(self, function : function) -> Simplex:
        """
        Purpose: Create transformed simplex by applying function to all vertices
        
        :param_data: function: Transformation function Rⁿ → Rⁿ
        
        :return: Simplex: New transformed simplex
        """
        return Simplex(
            np.stack([function(point) for point in self.points]),
            self.dim, self.order)
    
    def check_point(self, point : np.ndarray) -> bool:
        """
        Purpose: Check if point belongs to simplex using Carathéodory's theorem
        
        :param_data: point: Point to check
        
        :return: bool: True if point is in simplex
        """
        coeffs = np.vstack(
            [
                np.stack([point for point in self.points]).T, 
                np.ones(self.dim + 1)
            ]
        )

        try:
            check_num = np.linalg.solve(coeffs, np.append(point, 1))
        except np.linalg.LinAlgError:
            return False
        return (check_num >= -1e-15).all()
        
    def bisect(self) -> tuple:
        """
        Purpose: Bisect simplex by dividing longest edge
                 Returns two new smaller simplices
        
        :param_data: None
        
        :return: tuple: (simplex1, simplex2) - Resulting simplices
        """
        k, l = self.__calc_max_egde()
        middle_point = (self.points[k] + self.points[l]) / 2
        new_points = np.vstack([self.points, [middle_point]])
        simplex1_points = np.delete(new_points, k, axis=0)
        simplex2_points = np.delete(new_points, l, axis=0)
        return (
            Simplex(simplex1_points, self.dim, self.order), 
            Simplex(simplex2_points, self.dim, self.order),
        )
    
    def calc_diameter(self) -> float:
        """
        Purpose: Calculate simplex diameter as length of longest edge
        
        :param_data: None
        
        :return: float: Diameter value
        """
        max_len = 0
        for pair_index in itertools.combinations(range(self.order + 1), 2):
            length = self.norm(self.points[pair_index[0]] - self.points[pair_index[1]])
            if max_len < length:
                max_len = length
        return max_len
    
    @staticmethod
    def norm(point : np.ndarray):
        """
        Назначение: Вычисление длины радиус вектора 
        :param_data: None

        :return: norm
        """
        return np.sum(point ** 2) ** 0.5
    
    @staticmethod
    def inner_prod(point1 : np.ndarray, point2 : np.ndarray):
        """
        Purpose: Calculate inner product with another point's radius vector
        
        :param_data: point1: point for calculation
        :param_data: point2: point for calculation
        
        :return: float: Inner product value
        """
        return np.sum(point2 * point1)
    
    def calc_topological_degree(self, function : function, max_refinements : int = 5) -> float:
        """
        Purpose: Calculate topological degree using Kearfott's algorithm
                 With optional refinement iterations
        
        :param_data: function: Mapping function Rⁿ → Rⁿ
        :param_data: max_refinements: Maximum refinement iterations
        
        :return: float: Calculated topological degree
        """
        degree = 0
        faces = self.__generate_faces()
        for face in faces:
            sub_faces = [face[1]]
            for _ in range(max_refinements):
                face_contributions = [self.__calc_face_contribution(function, sub_face) for sub_face in sub_faces]
                if self.__needs_refinement(face_contributions):
                    sub_faces = list(itertools.chain(*[sub_face.bisect() for sub_face in sub_faces]))
                else:
                    degree += sum(face_contributions)
                    break
        return degree
    
    def __needs_refinement(self, contributions: list) -> bool:
        """
        Purpose: Determine if further refinement is needed for degree calculation
                 Checks if all contributions are zero (private helper method)
        
        :param_data: contributions: List of face contribution values
        
        :return: bool: True if refinement needed, False otherwise
        """
        return all(c == 0 for c in contributions)
    
    def __calc_face_contribution(self, func : function, face: tuple) -> int:
        """
        Purpose: Calculate topological degree contribution of a boundary face
                 Private method used in degree calculation
        
        :param_data: func: Mapping function Rⁿ → Rⁿ 
        :param_data: face: Boundary face simplex
        
        :return: int: Contribution value (1, -1 or 0)
        """
        face_vertices = face
        face_signs = np.ndarray((self.order, self.order), dtype=int)
        sgn = np.vectorize(lambda x : 1 if x >= 0 else 0)
        for i in range(self.order):
            for j in range(self.order):
                face_signs[i, j] = sgn(func(face_vertices.points[i])[j])
        par = self.__par(face_signs)
        return par
    
    def __calc_max_egde(self) -> tuple:
        """
        Purpose: Find indices of the longest edge in simplex
                 Private method used in bisection and reflection
        
        :param_data: None
        
        :return: tuple: (index1, index2) of edge vertices
        """
        max_len = 0
        edge = (0, 0)
        for pair_index in itertools.combinations(range(self.order + 1), 2):
            length = self.norm(self.points[pair_index[0]] - self.points[pair_index[1]])
            if max_len < length:
                max_len = length
                edge = pair_index
        return edge[0], edge[1]
    
    def __generate_faces(self) -> list:
        """
        Purpose: Generate all boundary faces of the simplex
                 Private method used in topological degree calculation
        
        :param_data: None
        
        :return: list: Tuples of (sign, face_simplex) for each boundary face
        """
        faces = []
        for i in range(0, self.order + 1):
            new_points = np.delete(self.points, i, axis=0)
            faces.append(((-1) ** i, Simplex(new_points, self.dim, self.order - 1)))
        return faces
    
    def __par(self, signs_matrix : np.ndarray) -> int:
        """
        Purpose: Calculate topological degree from sign matrix
                 Implements permutation checking (private helper method)
        
        :param_data: signs_matrix: Matrix of function sign evaluations
        
        :return: int: Degree contribution (1, -1 or 0)
        """
        mask_tril = np.tril(np.ones((self.order, self.order), dtype=bool))
        mask_diag = np.diag(np.ones(self.order - 1), k=1).astype(bool)
        check_lower = lambda perm : np.all(signs_matrix[perm][mask_tril] == 1)
        check_upper = lambda perm : np.all(signs_matrix[perm][mask_diag] == 0) if self.order > 2 else signs_matrix[perm][0, 1] == 0

        for perm in itertools.permutations(range(self.order)):
            perm = list(perm)
            if check_lower(perm) and check_upper(perm):
                swaps = 0
                for i in range(self.order):
                    if perm[i] != i:
                        for j in range(i, self.order):
                            if perm[j] == i:
                                perm[i], perm[j] = perm[j], perm[i]
                                swaps += 1
                                break
                if swaps % 2 == 0:
                    return 1
                return -1
        return 0

    def __str__(self):
        """
        Purpose: String representation of simplex vertices
        
        :param_data: None
        
        :return: str: Formatted vertex coordinates
        """
        stringed_simplex = ""
        for i, point in enumerate(self.points):
            stringed_simplex += f"point{i} : "
            stringed_simplex += str(point) + "\n"
        return stringed_simplex