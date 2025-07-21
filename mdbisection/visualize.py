from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .simplex import Simplex


def F_draw_2d(X : np.ndarray, Y : np.ndarray) -> tuple:
    """
    Purpose: Defines a 2D system of nonlinear equations for visualization
             f1(x,y) = x² - 4y
             f2(x,y) = y² - 2x + 4y
    
    :param_data: X: Meshgrid x-coordinates (np.ndarray)
    :param_data: Y: Meshgrid y-coordinates (np.ndarray)
    
    :return: tuple: (Z1, Z2) - Function values for both equations
    """
    Z1 = X ** 2 - 4 * Y
    Z2 = Y ** 2 - 2 * X + 4 * Y
    return Z1, Z2


def F_draw_3d(X : np.ndarray, Y : np.ndarray, Z : np.ndarray,) -> tuple:
    """
    Purpose: Defines a 3D system of nonlinear equations for visualization
             f1(x,y,z) = x² - 4y
             f2(x,y,z) = y² - 2x + 4y
             f3(x,y,z) = x² + y² - z² - 1
    
    :param_data: X: Meshgrid x-coordinates (np.ndarray)
    :param_data: Y: Meshgrid y-coordinates (np.ndarray)
    :param_data: Z: Meshgrid z-coordinates (np.ndarray)
    
    :return: tuple: (Z1, Z2, Z3) - Function values for all three equations
    """
    Z1 = X ** 2 - 4 * Y
    Z2 = Y ** 2 - 2 * X + 4 * Y
    Z3 = X ** 2 + Y ** 2 - Z ** 2 - 1
    return Z1, Z2, Z3


def test_draw_2d() -> None:
    """
    Purpose: Test function demonstrating 2D visualization of simplex algorithm
             Creates two sample simplices and visualizes them with F_draw_2d
    
    :param_data: None
    
    :return: None
    """
    simplex1 = Simplex(
            np.array([
                [0, 0], 
                [1, 0], 
                [0.5, 1]
            ]) ,2 ,2)
    simplex2 = Simplex(
            np.array([
                [1, 0], 
                [2, 0], 
                [1.5, 1]
            ]) ,2 ,2)
    draw_algorithm_2d(np.array([simplex1, simplex2]), F_draw_2d)


def test_draw_3d() -> None:
    """
    Purpose: Test function demonstrating 3D visualization of simplex algorithm
             Creates two sample simplices and visualizes them with F_draw_3d
    
    :param_data: None
    
    :return: None
    """
    simplex1 = Simplex(
        np.array([
            [0.2, 0.2, 0.2], 
            [-0.2, 0, 0], 
            [0, -0.2, 0], 
            [0, 0, -0.2], 
        ]), 3, 3)
    simplex2 = Simplex(
        np.array([
            [0.2, 0.2, 0.2], 
            ([-0.2, 0, 0] + [0.2, 0.2, 0.2]) / 2, 
            [0, -0.2, 0], 
            [0, 0, -0.2], 
        ]), 3, 3)
    draw_algorithm_3d(np.array([simplex1, simplex2]), F_draw_3d)


def draw_algorithm_3d(
        simplexes : list, 
        func : function, 
        range_x : tuple = (-3, 3), 
        range_y : tuple = (-3, 3), 
        range_z : tuple = (-3, 3)
    ) -> None:
    """
    Purpose: Visualizes 3D nonlinear system and simplex algorithm progress
             using isosurfaces and 3D simplex rendering
    
    :param_data: simplexes: List of Simplex objects to visualize
    :param_data: func: Function returning system equations (f1,f2,f3)
    :param_data: range_x: x-axis range for visualization (tuple)
    :param_data: range_y: y-axis range for visualization (tuple)
    :param_data: range_z: z-axis range for visualization (tuple)
    
    :return: None
    """

    x = np.linspace(range_x[0], range_x[1], 30)
    y = np.linspace(range_y[0], range_y[1], 30)
    z = np.linspace(range_z[0], range_z[1], 30)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Z1, Z2, Z3 = func(X, Y, Z)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    fig.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z1.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            opacity=0.5,
            colorscale='Magenta',
            showscale=False,
            caps=dict(x_show=True, y_show=True, z_show=True),  
        )
    )
    
    fig.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z2.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            opacity=0.5,
            colorscale='Greens',
            showscale=False,
            caps=dict(x_show=True, y_show=True, z_show=True),  
        )
    )
    
    fig.add_trace(
        go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z3.flatten(),
            isomin=0,
            isomax=0,
            surface_count=1,
            opacity=0.5,
            colorscale='Blues',
            showscale=False,
            caps=dict(x_show=True, y_show=True, z_show=True),  
        )
    )
    
    for tetrader in simplexes:
        points = tetrader.points
        
        edges =  [
            (0, 1), (0, 2), (0, 3),  
            (1, 2), (2, 3), (3, 1)  
        ]

        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=points[edge, 0],
                    y=points[edge, 1],
                    z=points[edge, 2],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False
                )
            )
        
        faces = [
            [0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]
        ]
        
        for face in faces:
            fig.add_trace(
                go.Mesh3d(
                    x=points[face, 0],
                    y=points[face, 1],
                    z=points[face, 2],
                    opacity=0.05,
                    color='purple',
                    flatshading=True,
                    showscale=False
                )
            )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-3, 3]),
            yaxis=dict(title='Y', range=[-3, 3]),
            zaxis=dict(title='Z', range=[-3, 3]),
            aspectmode='cube'
        ),
        width=800,
        height=600,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    fig.show()


def draw_optimize_3d(
        simplexes : list, 
        func : function, 
        range_x : tuple = (-5, 5), 
        range_y : tuple = (-5, 5)
    ) -> None:
    """
    Purpose: Visualizes optimization function and simplex algorithm progress
             in 3D space with surface plot and simplex rendering
    
    :param_data: simplexes: List of Simplex objects to visualize
    :param_data: func: Optimization function f(x,y) to plot
    :param_data: range_x: x-axis range for visualization (tuple)
    :param_data: range_y: y-axis range for visualization (tuple)
    
    :return: None
    """
    x = np.linspace(range_x[0], range_x[1], 100)
    y = np.linspace(range_y[0], range_y[1], 100)
    X, Y = np.meshgrid(x, y)

    Z1 = func(X, Y)

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
    
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=Z1,
            colorscale='Viridis',
            opacity=0.8,
            name='Функция'
        )
    )
    
    for tetrader in simplexes:
        points = tetrader.points
        
        edges =  [
            (0, 1), (0, 2), (0, 3),  
            (1, 2), (2, 3), (3, 1)  
        ]

        for edge in edges:
            fig.add_trace(
                go.Scatter3d(
                    x=points[edge, 0],
                    y=points[edge, 1],
                    z=points[edge, 2],
                    mode='lines',
                    line=dict(color='red', width=4),
                    showlegend=False
                )
            )
            
        faces = [
            [0, 1, 2], 
            [0, 1, 3], 
            [1, 2, 3], 
            [0, 2, 3]
        ]
        
        for face in faces:
            fig.add_trace(
                go.Mesh3d(
                    x=points[face, 0],
                    y=points[face, 1],
                    z=points[face, 2],
                    opacity=0.05,
                    color='red',
                    flatshading=True,
                    showscale=False
                )
            )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='f(X,Y)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        margin=dict(r=20, l=10, b=10, t=10)
    )
    
    fig.show()


def draw_algorithm_2d(simplexes : list, func : function, range_x : tuple = (-1, 1), range_y : tuple = (-1, 1)) -> None:
    """
    Purpose: Visualizes 2D nonlinear system and simplex algorithm progress
             using contour plots and filled simplex polygons
    
    :param_data: simplexes: List of Simplex objects to visualize
    :param_data: func: Function returning system equations (f1,f2)
    :param_data: range_x: x-axis range for visualization (tuple)
    :param_data: range_y: y-axis range for visualization (tuple)
    
    :return: None
    """
    delta = 0.025
    x = np.arange(range_x[0], range_x[1], delta)
    y = np.arange(range_y[0], range_y[1], delta)
    X, Y = np.meshgrid(x, y)
    Z1, Z2 = func(X, Y)
    fig, ax = plt.subplots()
    CS1 = ax.contour(X, Y, Z1, levels=[0], colors='r', linewidths=1)
    CS2 = ax.contour(X, Y, Z2, levels=[0], colors='b', linewidths=1)

    triangles = [
        [
            [point for point in point] 
            for point in simplex.points
        ] 
        for simplex in simplexes]
    for triangle in triangles:
        polygon = Polygon(triangle, closed=True, fill=True, alpha=0.5, edgecolor='m', facecolor='g', linewidth=1)
        ax.add_patch(polygon)

    ax.clabel(CS1, fontsize=10)
    ax.clabel(CS2, fontsize=10)
    ax.set_title('Изображение итераций алгоритма')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    plt.show()
