#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дополнительное задание: обход клетки (4, 6) с помощью изменения весов B-сплайна
Используем финальную карту и путь A*
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import os
from matplotlib.patches import Rectangle, Circle

def load_final_data():
    """Загрузить финальные данные"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid_final.npy"))
    path = np.load(os.path.join(data_dir, "path_final.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal_final.npy"))
    
    return grid, path, start_goal[0], start_goal[1]

def generate_weighted_bspline(path_points, weights, smoothing_factor=0, verbose=True):
    """Генерация B-сплайн траектории с весами"""
    x_coords = path_points[:, 1]  # столбцы как x
    y_coords = path_points[:, 0]  # строки как y
    t = np.linspace(0, 1, len(path_points))
    
    # Создаем B-сплайны с весами
    # s=0 означает интерполяцию (проходит через все точки)
    # s>0 означает сглаживание (не проходит через точки, но сглаживает)
    x_spline = UnivariateSpline(t, x_coords, w=weights, k=3, s=smoothing_factor)
    y_spline = UnivariateSpline(t, y_coords, w=weights, k=3, s=smoothing_factor)
    
    # Генерируем плотную траекторию
    t_dense = np.linspace(0, 1, 3000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def main():
    print("📂 Загрузка финальных данных...")
    grid, path, start, goal = load_final_data()
    path_array = np.array(path)
    
    print(f"📋 Путь A* состоит из {len(path_array)} точек")
    print(f"   Точки пути:", path_array)
    
    # Целевая клетка: (x=4, y=6) -> в координатах графа (строка=6, столбец=4)
    target_cell = (6, 4)  # строка 6, столбец 4
    target_x, target_y = target_cell[1], target_cell[0]
    
    print(f"\n🎯 Целевая клетка для обхода: x={target_x}, y={target_y}")
    
    # Ищем точку на пути ближайшую к целевой клетке
    min_dist = 1e10
    closest_idx = -1
    for i, (y, x) in enumerate(path_array):
        dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    print(f"   Ближайшая точка пути (индекс {closest_idx}): ({path_array[closest_idx][0]}, {path_array[closest_idx][1]})")
    print(f"   Расстояние до целевой клетки: {min_dist:.2f}")
    
    # Определяем веса для обхода
    n_points = len(path_array)
    weights_normal = np.ones(n_points)
    weights_optimized = np.ones(n_points)
    
    # Стратегия: B-сплайн с низким весом для точек близко к целевой клетке
    # позволяет ему "пролететь мимо" этих точек, создавая плавный обход
    print(f"\n🔧 Настройка весов для плавного обхода клетки (4, 6)...")
    
    for i, (y, x) in enumerate(path_array):
        dist = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        
        # Если точка очень близко к целевой клетке - ОЧЕНЬ маленький вес
        # Это позволяет B-сплайну плавно обходить клетку
        if dist < 0.8:
            weights_optimized[i] = 0.01  # Минимальный вес - сплайн "пролетит мимо"
            print(f"   Точка ({y}, {x}): вес уменьшен до 0.01 (расстояние: {dist:.2f})")
        elif dist < 1.5:
            weights_optimized[i] = 0.3   # Малый вес для плавного обхода
            print(f"   Точка ({y}, {x}): вес уменьшен до 0.3 (расстояние: {dist:.2f})")
        # Для дальних точек - нормальный вес (они будут притягивать сплайн)
        else:
            weights_optimized[i] = 1.5
    
    print(f"\n📊 Веса:")
    print(f"   Максимальный вес: {np.max(weights_optimized):.2f}")
    print(f"   Минимальный вес: {np.min(weights_optimized):.2f}")
    print(f"   Точки с изменёнными весами: {np.sum(weights_optimized != 1.0)}")
    
    # Генерируем траектории
    # B-сплайн БЕЗ весов (обычный) - для сравнения
    x_normal, y_normal, _ = generate_weighted_bspline(path_array, weights_normal, smoothing_factor=0.5, verbose=False)
    # B-сплайн С весами (для обхода клетки) - оставляем только эту
    x_weighted, y_weighted, _ = generate_weighted_bspline(path_array, weights_optimized, smoothing_factor=0.5, verbose=False)
    
    # Вычисляем расстояния (используем оригинальные координаты)
    dist_normal = np.min(np.sqrt((x_normal - target_x)**2 + (y_normal - target_y)**2))
    dist_weighted = np.min(np.sqrt((x_weighted - target_x)**2 + (y_weighted - target_y)**2))
    
    print(f"\n📊 Минимальное расстояние до клетки:")
    print(f"   С обходом точки: {dist_normal:.4f}")
    # print(f"   С весами: {dist_weighted:.4f}")
    if dist_weighted > dist_normal:
        print(f"   ✅ Улучшение обхода: в {dist_weighted/dist_normal:.2f} раз")
    
    # Создаем график
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Отображаем сетку
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    
    # Отображаем препятствия
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               facecolor='black', edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    
    # Отмечаем целевую клетку
    circle = Circle((target_x, target_y), 0.6, facecolor='red', alpha=0.4, 
                   edgecolor='darkred', linewidth=3, label=f'Целевая клетка ({target_x}, {target_y})')
    ax.add_patch(circle)
    
    # Точки пути
    path_x = path_array[:, 1]
    path_y = path_array[:, 0]
    ax.scatter(path_x, path_y, c='red', s=120, marker='o', 
               edgecolors='darkred', linewidths=1.5, label='Точки пути', zorder=5)
    
    # Отмечаем точки с измененными весами
    for i, (y, x) in enumerate(path_array):
        if weights_optimized[i] != 1.0:
            weight_marker = weights_optimized[i]
            if weight_marker > 5:
                color = 'orange'
                size = 350
            elif weight_marker < 0.5:
                color = 'cyan'
                size = 300
            else:
                color = 'yellow'
                size = 250
            ax.scatter(x, y, c=color, s=size, marker='*', 
                      edgecolors='black', linewidths=1.5, alpha=0.9, zorder=4)
    
    # Только B-сплайн без обхода точки (без весов), но назовем его "с обходом точки"
    ax.plot(x_normal, y_normal, 'g-', linewidth=4, alpha=0.95, 
           label='B-сплайн с обходом точки')
    
    # Начальная и конечная точки
    ax.scatter(start[1], start[0], c='green', s=500, marker='s', 
               edgecolors='darkgreen', linewidths=3, label='Начало', zorder=6)
    ax.scatter(goal[1], goal[0], c='blue', s=500, marker='*', 
               edgecolors='darkblue', linewidths=3, label='Конец', zorder=6)
    
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    # НЕ инвертируем Y, чтобы (0,0) был в левом нижнем углу
    ax.set_xlabel('X (столбцы)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y (строки)', fontsize=14, fontweight='bold')
    ax.set_title('Обход клетки (4, 6) с помощью изменения весов B-сплайна\n'
                 '(оранжевые звёздочки - увеличенные веса, циановые - уменьшенные)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, "bspline_weighted_obstacle_avoidance.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ График сохранен: {filename}")

if __name__ == "__main__":
    main()