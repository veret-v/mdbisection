from __future__ import annotations

import copy
import itertools

import numpy as np


class Point:
    """
    Purpose: Represents a point in n-dimensional space with coordinates and norm
            Provides vector operations and coordinate management
    """
        
    _coords : np.ndarray
    _dim    : int
    norm    : float

    def __init__(self, coords : np.ndarray, dim : int) -> None:
        """
        Purpose: Represents a point in n-dimensional space with coordinates and norm
                Provides vector operations and coordinate management
        
        :param_data: coords: Coordinate values as numpy array
        :param_data: dim: Dimension of the space
        
        :return: Point instance
        """
        self._dim = dim
        self._coords = coords
        self.norm = self.__norm()

    @property
    def dim(self):
        """
        Purpose: Get or set the dimension of the point's space
                 Validates dimension consistency when setting coordinates
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current dimension
        """
        return self._dim
    
    @dim.setter
    def dim(self, dim):
        """
        Purpose: Get or set the dimension of the point's space
                 Validates dimension consistency when setting coordinates
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current dimension
        """
        self._dim = dim

    @property
    def coords(self):
        """
        Purpose: Get or set the point's coordinates
                 Validates coordinate length matches dimension
        
        :param_data: None (getter) or coords (setter)
        
        :return: np.ndarray: Current coordinates
        """

        return self._coords
    
    @coords.setter
    def coords(self, coords):
        """
        Purpose: Get or set the point's coordinates
                 Validates coordinate length matches dimension
        
        :param_data: None (getter) or coords (setter)
        
        :return: np.ndarray: Current coordinates
        """

        if self.dim != len(coords):
            raise ValueError(f"Количество координат({coords}) не совпадает с размерностью({self.dim})")
        self._coords = coords.copy()

    def inner_prod(self, point : Point) -> float:
        """
        Purpose: Calculate inner product with another point's radius vector
        
        :param_data: point: Other point for calculation
        
        :return: float: Inner product value
        """
        return np.sum(self.coords * point.coords)

    def __sub__(self, point : Point) -> Point:
        """
        Purpose: Subtract another point's radius vector
        
        :param_data: point: Other point for subtraction
        
        :return: Point: Resulting point
        """
        return Point(self.coords - point.coords, self.dim)
    
    def __add__(self, point : Point) -> Point:
        """
        Purpose: Add another point's radius vector
        
        :param_data: point: Other point for addition
        
        :return: Point: Resulting point
        """
        return Point(self.coords + point.coords, self.dim)
    
    def __mul__(self, val : float) -> Point:
        """
        Purpose: Multiply radius vector by scalar value
        
        :param_data: val: Scalar multiplier
        
        :return: Point: Scaled point
        """
        return Point(val * self.coords, self.dim)
    
    def __truediv__(self, val : float) -> Point:
        """
        Purpose: Divide radius vector by scalar value
        
        :param_data: val: Scalar divisor
        
        :return: Point: Scaled point
        """
        return Point(self.coords / val, self.dim)
    
    def __norm(self) -> float:
        """
        Назначение: Вычисление длины радиус вектора 
        :param_data: None

        :return: norm
        """
        return np.sum(self.coords ** 2) ** 0.5


class Simplex:
    """
    Purpose: Represents a geometric simplex with vertices in n-dimensional space
            Provides operations like reflection, bisection, and topological degree calculation
    """
    _points : list[Point]
    _dim    : int
    _order  : int

    def __init__(self, points : list[Point], dim : int, order : int) -> None:
        """
        Purpose: Represents a geometric simplex with vertices in n-dimensional space
                Provides operations like reflection, bisection, and topological degree calculation
        
        :param_data: points: List of vertex Points
        :param_data: dim: Space dimension
        :param_data: order: Simplex order (number of vertices - 1)
        
        :return: Simplex instance
        """
        self._points = points
        self._dim = dim
        self._order = order

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
    def dim(self, points):
        """
        Purpose: Get or set simplex vertices
                 Maintains deep copies when setting
        
        :param_data: None (getter) or points (setter)
        
        :return: list[Point]: Current vertices
        """
        self._dim = copy.deepcopy(points)

    @property
    def dim(self):
        """
        Purpose: Get or set space dimension
        
        :param_data: None (getter) or dim (setter)
        
        :return: int: Current dimension
        """
        return self._dim
    
    @dim.setter
    def dim(self, dim):
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
    def order(self, order):
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
        return np.array([np.sum(func(point.coords)**2) ** 0.5 for point  in  self.points])
    
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
                    [self.points[p] for p in (set(range(self.order + 1)) - set([0]))], 
                    self.dim,
                    self.order
                ).transform(func).distance(Point(np.zeros(self.order), self.order))
                i0 = iterable_index[0]

                for i in iterable_index:
                    dist = Simplex(
                        [self.points[p] for p in (set(range(self.order + 1)) - set([i]))], 
                        self.dim,
                        self.order
                    ).transform(func).distance(Point(np.zeros(self.order), self.order))

                    if min_dist > dist:
                        min_dist = dist
                        i0 = i
            else:
                v = Point(func(self.points[k].coords) - func(self.points[l].coords), self.dim)
                fd = Point(func(middle_point.coords) - (func(self.points[k].coords) + func(self.points[l].coords)) / 2, self.dim)
                w  = Point(func(self.points[i0].coords) - (func(self.points[k].coords) + func(self.points[l].coords)) / 2, self.dim)
            
                projection = lambda x : x - v * x.inner_prod(v) / v.inner_prod(v) 

                i0 = iterable_index[0]
                max_metric = projection(w).inner_prod(projection(fd)) / (projection(w).norm * projection(fd).norm)

                for i  in iterable_index:
                    w  = Point(func(self.points[i].coords) - (func(self.points[k].coords) + func(self.points[l].coords)) / 2, self.dim)
                    metric = projection(w).inner_prod(projection(fd)) / (projection(w).norm * projection(fd).norm)
                    if metric > max_metric:
                        max_metric = metric
                        i0 = i

        reflected_point = self.points[k] + self.points[l] - self.points[i0]
        flag = self.check_point(reflected_point)

        new_points = self.points[:]
        new_points.append(middle_point)
        new_points.append(reflected_point)
        new_points.pop(i0)
        simplex1_points = new_points[:]
        simplex2_points = new_points[:]
        simplex1_points.pop(k)
        simplex2_points.pop(l)
        return (
            Simplex(simplex1_points, self.dim, self.order), 
            Simplex(simplex2_points, self.dim, self.order),
            flag
        )
    
    def distance(self, target_point : Point) -> float:
        """
        Purpose: Calculate minimal distance to target point
                 Uses metric min(||Xi - Y||) where Xi are vertices
        
        :param_data: target_point: Point to measure distance to
        
        :return: float: Minimal distance value
        """
        min_dist = (self.points[0] - target_point).norm
        for point in self.points:
            dist = (point - target_point).norm
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
            [Point(function(point.coords), self.dim) for point in self.points],
            self.dim, self.order)
    
    def check_point(self, point : Point) -> bool:
        """
        Purpose: Check if point belongs to simplex using Carathéodory's theorem
        
        :param_data: point: Point to check
        
        :return: bool: True if point is in simplex
        """
        coeffs = np.vstack(
            [
                np.stack([point.coords for point in self.points]).T, 
                np.ones(self.dim + 1)
            ]
        )

        try:
            check_num = np.linalg.solve(coeffs, np.append(point.coords, 1))
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
        new_points = self.points[:]
        new_points.append(middle_point)
        simplex1_points = new_points[:]
        simplex2_points = new_points[:]
        simplex1_points.pop(k)
        simplex2_points.pop(l)
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
            length = (self.points[pair_index[0]] - self.points[pair_index[1]]).norm
            if max_len < length:
                max_len = length
        return max_len
    
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
                face_signs[i, j] = sgn(func(face_vertices.points[i].coords)[j])
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
            length = (self.points[pair_index[0]] - self.points[pair_index[1]]).norm
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
            new_points = self.points[:]
            new_points.pop(i)
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
            stringed_simplex += str(point.coords) + "\n"
        return stringed_simplex