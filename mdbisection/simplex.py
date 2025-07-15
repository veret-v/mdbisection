from __future__ import annotations

import math
import itertools

import numpy as np


class Point:
    """Класс точки"""
    coords : np.ndarray
    dim    : int
    norm   : float

    def __init__(self, coords : np.ndarray, dim : int) -> None:
        """
        Назначение: Инициализация точки
        :param_data: coords: Координаты
        :param_data: dim: Размерность пространства

        :return: None
        """
        if dim != len(coords):
            raise ValueError(f"Количество координат({coords}) не совпадает с размерностью({dim})")
        
        self.dim = dim
        self.coords = coords.copy()  
        self.norm = self.__norm()
        
    def inner_prod(self, point : Point) -> float:
        """
        Назначение: Вычисление скалярного произведения радиус векторов
        :param_data: point: Радиус вектор другой точки

        :return: inner_product
        """
        return np.sum(self.coords * point.coords)

    def __sub__(self, point : Point) -> Point:
        """
        Назначение: Вычитание радиус векторов
        :param_data: point: Радиус вектор другой точки

        :return: sub
        """
        return Point(self.coords - point.coords, self.dim)
    
    def __add__(self, point : Point) -> Point:
        """
        Назначение: Сложение радиус векторов
        :param_data: point: Радиус вектор другой точки

        :return: sum
        """
        return Point(self.coords + point.coords, self.dim)
    
    def __mul__(self, val : float) -> Point:
        """
        Назначение: умнжение радиус вектора на число
        :param_data: val: Действительное число

        :return: prod
        """
        return Point(val * self.coords, self.dim)
    
    def __truediv__(self, val : float) -> Point:
        """
        Назначение: Деление радиус вектора на число
        :param_data: val: Действительное число

        :return: div
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
    """Класс симплекса"""
    points : list[Point]
    dim    : int
    order  : int
    center : Point

    def __init__(self, points : list[Point], dim : int, order : int) -> None:
        """
        Назначение: Инициализация симплекса
        :param_data: points: Вершины симплекса 
        :param_data: dim: Размерность пространства
        :param_data: order: Порядок симплекса

        :return: None
        """
        self.points = points.copy()
        self.dim = dim
        self.order = order
        self.center = self.__calc_center()

    def calc_point_norms(self, func : function) -> np.ndarray:
        """
        Назначение: Вычисление норм радиус векторов точек
        :param_data: func: Функция f : R * n -> R * n

        :return: norm: Норма радиус вектора
        """
        return np.array([np.sum(func(point.coords)**2) ** 0.5 for point  in  self.points])
    
    def reflection(self,  func : function) -> tuple:
        """
        Назначение: Создание новых симплексов путем отражения
        :param_data: func: Функция f : R * n -> R * n

        :return: simplex1, simplex2: Новые симплексы
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
        Назначение: Вычисление расстояния до симплекса по метрике 
                    dist=min(||Xi - Y||), где Xi вершины симплекса
        :param_data: target_point: Точка до которой вычисляется рассстояние

        :return: min_dist: Минимальное расстояние от точки до симплекса
        """
        min_dist = (self.points[0] - target_point).norm
        for point in self.points:
            dist = (point - target_point).norm
            if min_dist > dist:
                min_dist = dist
        return min_dist

    def transform(self, function : function) -> Simplex:
        """
        Назначение: Преобразование симплекса под действием функции 
        :param_data: function: Функция f : R * n -> R * n
        
        :return: simplex: Симплекс с измененными координатами
        """
        return Simplex(
            [Point(function(point.coords), self.dim) for point in self.points],
            self.dim, self.order)
    
    def check_point(self, point : Point) -> bool:
        """
        Назначение: Проверка принадлежности точки симплексу, 
                    используя теорему Каратеодори.
        :param_data: point: Проверяемая точка

        :return: bool: Истинность принадлежности
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
        
        return (check_num >= 0).all()
        
    def bisect(self) -> tuple:
        """
        Назначение: Разделение симплекса на два путем 
                    деления пополам наибольшей стороны
        :param_data: None
        
        :return: tuple(simplex1, simplex2)
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
        Назначение: Вычисление диаметра симплекса, как наибольшей его грани
        :param_data: None
        
        :return: max_len
        """
        max_len = 0
        for pair_index in itertools.combinations(range(self.order + 1), 2):
            length = (self.points[pair_index[0]] - self.points[pair_index[1]]).norm
            if max_len < length:
                max_len = length
        return max_len
    
    def calc_topological_degree(self, function : function, max_refinements : int = 5) -> float:
        """
        Назначение: Вычисление степени отображения по алгоритму Кеарфотта
        :param_data: function: Функция f : R * n -> R * n
        :param_data: max_refinements: Количество уточняющих итераций
        
        :return: degree: степень отображения
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
        Назначение: Проверка на необходимость дольнейшего 
                    уточнения степени отображения
        :param_data: contributions: Список степеней отображения для подсимплексов границы
        
        :return: bool
        """
        return all(c == 0 for c in contributions)
    
    def __calc_face_contribution(self, func : function, face: tuple) -> int:
        """
        Назначение: Вычисление степени отображения симплекса границы
        :param_data: func: Функция f : R * n -> R * n
        :param_data: face: Симплекс границы
        
        :return: par
        """
        face_vertices = face
        face_signs = np.ndarray((self.order, self.order), dtype=int)
        sgn = np.vectorize(lambda x : 1 if x >= 0 else 0)
        for i in range(self.order):
            for j in range(self.order):
                face_signs[i, j] = sgn(func(face_vertices.points[i].coords)[j])
        # print(face_signs)
        par = self.__par(face_signs)
        return par
    
    def __calc_max_egde(self) -> tuple:
        """
        Назначение: Вычисление индексов ребра максимальной длины
        :param_data: None
        
        :return: indexes
        """
        max_len = 0
        edge = (0, 0)
        for pair_index in itertools.combinations(range(self.order + 1), 2):
            length = (self.points[pair_index[0]] - self.points[pair_index[1]]).norm
            if max_len < length:
                max_len = length
                edge = pair_index
        return edge[0], edge[1]
    
    def __calc_center(self) -> Point:
        """
        Назначение: Вычисление центра массы симплекса
        :param_data: None
        
        :return: point
        """
        center = Point(np.zeros(self.dim), self.dim)
        for point in self.points:
            center = center + point
        return center / (self.order + 1)
        
    def __generate_faces(self) -> list:
        """
        Назначение: Генерация массива симплексов границы
        :param_data: None
        
        :return: list
        """
        faces = []
        for i in range(0, self.order + 1):
            new_points = self.points[:]
            new_points.pop(i)
            faces.append(((-1) ** i, Simplex(new_points, self.dim, self.order - 1)))
        return faces
    
    def __par(self, signs_matrix : np.ndarray) -> int:
        """
        Назначение: Вычисление степени отображения оосновываясь на матрице знаков
        :param_data: signs_matrix: матрица знаков получаемая применением 
                                   функции к каждой точке граничных симплексов
        
        :return: par: степень отображения
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
        stringed_simplex = ""
        for i, point in enumerate(self.points):
            stringed_simplex += f"point{i} : "
            stringed_simplex += str(point.coords) + "\n"
        return stringed_simplex