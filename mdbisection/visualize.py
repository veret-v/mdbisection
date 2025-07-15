from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from .simplex import Simplex, Point


def F_draw_2d(X : np.ndarray, Y : np.ndarray) -> tuple:
    Z1 = X ** 2 - 4 * Y
    Z2 = Y ** 2 - 2 * X + 4 * Y
    return Z1, Z2


def F_draw_3d(X : np.ndarray, Y : np.ndarray, Z : np.ndarray,) -> tuple:
    Z1 = X ** 2 - 4 * Y
    Z2 = Y ** 2 - 2 * X + 4 * Y
    Z3 = X ** 2 + Y ** 2 - Z ** 2 - 1
    return Z1, Z2, Z3


def test_draw_2d() -> None:
    simplex1 = Simplex(
            [
                Point(np.array([0, 0]), 2), 
                Point(np.array([1, 0]), 2), 
                Point(np.array([0.5, 1]), 2)
            ]
        ,2 ,2)
    simplex2 = Simplex(
            [
                Point(np.array([1, 0]), 2), 
                Point(np.array([2, 0]), 2), 
                Point(np.array([1.5, 1]), 2)
            ]
        ,2 ,2)
    draw_algorithm_2d(np.array([simplex1, simplex2]), F_draw_2d)


def test_draw_3d() -> None:
    simplex1 = Simplex(
        [
            Point(np.array([0.2, 0.2, 0.2]), 3), 
            Point(np.array([-0.2, 0, 0]), 3), 
            Point(np.array([0, -0.2, 0]), 3), 
            Point(np.array([0, 0, -0.2]), 3), 
        ], 3, 3)
    simplex2 = Simplex(
        [
            Point(np.array([0.2, 0.2, 0.2]), 3), 
            (Point(np.array([-0.2, 0, 0]), 3) + Point(np.array([0.2, 0.2, 0.2]), 3)) / 2, 
            Point(np.array([0, -0.2, 0]), 3), 
            Point(np.array([0, 0, -0.2]), 3), 
        ], 3, 3)
    draw_algorithm_3d(np.array([simplex1, simplex2]), F_draw_3d)


def draw_algorithm_3d(simplexes : list, func : function) -> None:
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    z = np.linspace(-3, 3, 30)
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
        points = np.array([[coord for coord in point.coords] for point in tetrader.points])
        
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='lines',
                opacity=1,
                marker=dict(size=6, color='red'),
                showlegend=False,
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


def draw_algorithm_2d(simplexes : list, func : function) -> None:
    delta = 0.025
    x = np.arange(-10.0, 10.0, delta)
    y = np.arange(-10.0, 10.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1, Z2 = func(X, Y)
    fig, ax = plt.subplots()
    CS1 = ax.contour(X, Y, Z1, levels=[0], colors='r', linewidths=1)
    CS2 = ax.contour(X, Y, Z2, levels=[0], colors='b', linewidths=1)

    triangles = [
        [
            [coord for coord in point.coords] 
            for point in simplex.points
        ] 
        for simplex in simplexes]
    for triangle in triangles:
        polygon = Polygon(triangle, closed=True, fill=True, alpha=0.5, edgecolor='m', facecolor='g', linewidth=1)
        ax.add_patch(polygon)

    ax.clabel(CS1, fontsize=10)
    ax.clabel(CS2, fontsize=10)
    ax.set_title('Изображение итераций алгоритма')
    ax.grid()
    plt.show()
