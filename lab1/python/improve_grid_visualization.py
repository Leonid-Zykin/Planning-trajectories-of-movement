import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline

def load_final_data():
    """Загрузить финальные данные"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid_final.npy"))
    path = np.load(os.path.join(data_dir, "path_final.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal_final.npy"))
    
    return grid, path, start_goal[0], start_goal[1]

def generate_c0_trajectory(path_points):
    """Генерация C⁰-гладкой траектории (кусочно-линейная)"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    
    t = np.linspace(0, 1, len(path_points))
    x_interp = interp1d(t, x_coords, kind='linear')
    y_interp = interp1d(t, y_coords, kind='linear')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_interp(t_dense)
    y_traj = y_interp(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c1_trajectory(path_points):
    """Генерация C¹-гладкой траектории"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='natural')
    y_spline = CubicSpline(t, y_coords, bc_type='natural')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c2_trajectory(path_points):
    """Генерация C²-гладкой траектории"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='clamped')
    y_spline = CubicSpline(t, y_coords, bc_type='clamped')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_bspline_trajectory(path_points, smoothing_factor=None):
    """Генерация B-сплайн траектории"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    if smoothing_factor is None:
        x_spline = UnivariateSpline(t, x_coords, k=3)
        y_spline = UnivariateSpline(t, y_coords, k=3)
    else:
        x_spline = UnivariateSpline(t, x_coords, s=smoothing_factor, k=3)
        y_spline = UnivariateSpline(t, y_coords, s=smoothing_factor, k=3)
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def plot_map_with_trajectories_improved(grid: np.ndarray, path: list, 
                                       start: tuple, goal: tuple,
                                       x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline,
                                       filename: str):
    """Построение карты со всеми траекториями с улучшенной сеткой"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Отображаем карту
    ax.imshow(grid, cmap='binary', origin='lower', extent=[-0.5, grid.shape[1]-0.5, -0.5, grid.shape[0]-0.5])
    
    # Добавляем четкую сетку 10x10
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=1, alpha=0.7)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=1, alpha=0.7)
    
    # Добавляем подписи координат
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_xticklabels(range(grid.shape[1]), fontsize=10)
    ax.set_yticklabels(range(grid.shape[0]), fontsize=10)
    
    # Отображаем траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=3, label='C⁰-гладкая', alpha=0.8)
    ax.plot(x_c1, y_c1, 'g-', linewidth=3, label='C¹-гладкая', alpha=0.8)
    ax.plot(x_c2, y_c2, 'm-', linewidth=3, label='C²-гладкая', alpha=0.8)
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=3, label='B-сплайн', alpha=0.8)
    
    # Отображаем путь A*
    if len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r--', linewidth=2, alpha=0.7, label='Путь A*')
        ax.scatter(path_array[:, 1], path_array[:, 0], c='red', s=40, zorder=5, alpha=0.8)
    
    # Отображаем начальную и конечную точки
    ax.scatter(start[1], start[0], c='green', s=300, marker='s', label='Начало', zorder=10, edgecolors='black', linewidth=2)
    ax.scatter(goal[1], goal[0], c='blue', s=300, marker='*', label='Конец', zorder=10, edgecolors='black', linewidth=2)
    
    # Настройки осей
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_xlabel('X (столбцы)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (строки)', fontsize=12, fontweight='bold')
    ax.set_title('Бинарная карта 10×10 с траекториями', fontsize=14, fontweight='bold')
    
    # Улучшенная легенда
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Добавляем текст с информацией о карте
    info_text = f'Размер карты: {grid.shape[0]}×{grid.shape[1]}\nПрепятствий: {np.sum(grid)}\nПуть: {len(path)} ячеек'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectories_comparison_improved(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, 
                                        x_bspline, y_bspline, path_points, filename):
    """Построение сравнения траекторий с улучшенной сеткой"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Отображаем траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=3, label='C⁰-гладкая', alpha=0.8)
    ax.plot(x_c1, y_c1, 'g-', linewidth=3, label='C¹-гладкая', alpha=0.8)
    ax.plot(x_c2, y_c2, 'm-', linewidth=3, label='C²-гладкая', alpha=0.8)
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=3, label='B-сплайн', alpha=0.8)
    
    # Отображаем точки пути A*
    ax.scatter(path_points[:, 1], path_points[:, 0], c='red', s=80, 
               label='Точки пути A*', zorder=5, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение траекторий с разной гладкостью', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple_bsplines_improved(path_points, smoothing_factors, filename):
    """Построение нескольких B-сплайн траекторий с улучшенной визуализацией"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['orange', 'red', 'purple', 'brown', 'pink']
    
    for i, s in enumerate(smoothing_factors):
        x_bspline, y_bspline, _ = generate_bspline_trajectory(path_points, s)
        label = f'B-сплайн (s={s})' if s is not None else 'B-сплайн (авто)'
        ax.plot(x_bspline, y_bspline, color=colors[i % len(colors)], 
                linewidth=3, label=label, alpha=0.8)
    
    # Отображаем точки пути A*
    ax.scatter(path_points[:, 1], path_points[:, 0], c='red', s=80, 
               label='Точки пути A*', zorder=5, edgecolors='black', linewidth=1)
    
    # Добавляем сетку
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_title('B-сплайн траектории с разными параметрами сглаживания', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка финальных данных...")
    grid, path, start, goal = load_final_data()
    
    print("Генерация траекторий...")
    path_array = np.array(path)
    
    # Генерируем траектории
    x_c0, y_c0, t_c0 = generate_c0_trajectory(path_array)
    x_c1, y_c1, t_c1 = generate_c1_trajectory(path_array)
    x_c2, y_c2, t_c2 = generate_c2_trajectory(path_array)
    x_bspline, y_bspline, t_bspline = generate_bspline_trajectory(path_array)
    
    print("Создание улучшенных графиков...")
    
    # Карта со всеми траекториями с улучшенной сеткой
    plot_map_with_trajectories_improved(grid, path, start, goal,
                                      x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline,
                                      "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1/astar_path_with_trajectories_final.png")
    
    plot_trajectories_comparison_improved(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, 
                                        x_bspline, y_bspline, path_array,
                                        "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/trajectories_comparison_final.png")
    
    # Строим несколько B-сплайн траекторий с разными параметрами
    smoothing_factors = [None, 0.1, 0.5, 1.0, 2.0]
    plot_multiple_bsplines_improved(path_array, smoothing_factors,
                                   "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/multiple_bsplines_final.png")
    
    print("Улучшенные графики сохранены!")

if __name__ == "__main__":
    main()
