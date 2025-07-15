from __future__ import annotations

import copy

import numpy as np

from mdbisection.simplex import Simplex, Point


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
    Назначение: Данная функция, возвращает алгоритм на этап подсчета степени 
                отображения, в случае если алгоритм не достиг точности за предельное 
                число итераций, либо в случае если более тривиальные условия 
                проверки принадлежности симплексу оказлись невалидны.
    :param_data: working_simplex
    :param_data: simplexes
    :param_data: func
    :param_data: max_diameter
    :param_data: epsilon1
    :param_data: m
    :param_data: max_num_approx
    :param_data: dim

    :return: simplex: Корректный симплекс
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
    Назначение: Поиск решения обобщенным методом бисекции 
                для определенных систем нелинейных уравнений
    :param_data: func: Функция f : R * n -> R * n
    :param_data: init_simplex: Начальный симплекс
    :param_data: epsilon1
    :param_data: epsilon2
    :param_data: dim: Размерность пространства
    :param_data: max_num_approx: Максимальное количество симплексов 
                                перед возвращением к стадии 
                                определния существования решения

    :return: simplexes: Массив сходящихся симплексов
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
        return simplexes

    while(working_simplex.calc_diameter() > epsilon1 and (working_simplex.calc_point_norms(func) > epsilon2).any()):
        if m == max_num_approx:
            working_simplex, m = top_deg_stage(
                working_simplex, simplexes, func, 
                max_diameter, epsilon1, m, max_num_approx, dim
            )
            max_diameter = working_simplex.calc_diameter()
            print(1)
        else:
            simplex1, simplex2 = working_simplex.bisect()
            if simplex1.transform(func).check_point(Point(np.zeros(dim), dim)):
                m += 1
                simplexes.append(simplex1)
                working_simplex = simplex1
                print(2)
            elif simplex2.transform(func).check_point(Point(np.zeros(dim), dim)):
                m += 1
                simplexes.append(simplex2)
                working_simplex = simplex2
                print(3)
            else:
                refl_simplex1, refl_simplex2, flag = working_simplex.reflection(func)
                if refl_simplex1.transform(func).check_point(Point(np.zeros(dim), dim)) and flag:
                    m += 1
                    simplexes.append(refl_simplex1)
                    working_simplex = refl_simplex1
                    print(4)
                elif refl_simplex2.transform(func).check_point(Point(np.zeros(dim), dim)) and flag:
                    m += 1
                    simplexes.append(refl_simplex2)
                    working_simplex = refl_simplex2
                    print(5)
                else:
                    working_simplex, m = top_deg_stage(
                        working_simplex, simplexes, func, 
                        max_diameter, epsilon1, m, max_num_approx, dim
                    )
                    max_diameter = working_simplex.calc_diameter()
                    print(6)
        print(working_simplex.check_point(Point(np.array([0.1, 0.1, 0.1]), dim)))
        output_simplexes.append(working_simplex)   
    return output_simplexes

