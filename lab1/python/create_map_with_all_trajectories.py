import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline

def load_modified_data():
    """Загрузить модифицированные данные"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid_modified_for_bspline.npy"))
    path = np.load(os.path.join(data_dir, "path_modified.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal_modified.npy"))
    
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

def b_spline_smoothing(path_points, s=None):
    """Сглаживание траектории с помощью B-сплайна"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = UnivariateSpline(t, x_coords, s=s, k=3)
    y_spline = UnivariateSpline(t, y_coords, s=s, k=3)
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def plot_map_with_all_trajectories(grid, path, start, goal, filename):
    """Построить карту со всеми траекториями"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем сетку
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    
    # Отображаем препятствия
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               facecolor='black', edgecolor='black', alpha=0.7)
                ax.add_patch(rect)
    
    # Генерируем все траектории
    path_array = np.array(path)
    
    print("Генерация всех траекторий...")
    x_c0, y_c0, _ = generate_c0_trajectory(path_array)
    x_c1, y_c1, _ = generate_c1_trajectory(path_array)
    x_c2, y_c2, _ = generate_c2_trajectory(path_array)
    x_bspline, y_bspline, _ = b_spline_smoothing(path_array)
    
    # Отображаем исходные точки пути A*
    ax.scatter(path_array[:, 1], path_array[:, 0], c='red', s=100, 
               label='Точки пути A*', zorder=5)
    
    # Отображаем все траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=2, label='C⁰-гладкая (линейная)')
    ax.plot(x_c1, y_c1, 'g-', linewidth=2, label='C¹-гладкая (кубический сплайн)')
    ax.plot(x_c2, y_c2, 'm-', linewidth=2, label='C²-гладкая (кубический сплайн)')
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=3, label='B-сплайн сглаживание')
    
    # Отмечаем начальную и конечную точки
    ax.scatter(start[1], start[0], c='green', s=200, marker='s', 
               label='Начальная точка', zorder=6)
    ax.scatter(goal[1], goal[0], c='blue', s=200, marker='*', 
               label='Конечная точка', zorder=6)
    
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X (столбцы)')
    ax.set_ylabel('Y (строки)')
    ax.set_title('Бинарная карта и траектории с разной гладкостью')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка модифицированных данных...")
    grid, path, start, goal = load_modified_data()
    
    print("Создание изображения карты со всеми траекториями...")
    
    # Сохраняем карту со всеми траекториями
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_map_with_all_trajectories(grid, path, start, goal, 
                                 os.path.join(output_dir, "astar_path_with_trajectories.png"))
    
    print(f"Изображение сохранено в {output_dir}")
    print("Создана карта со всеми траекториями!")

if __name__ == "__main__":
    main()
