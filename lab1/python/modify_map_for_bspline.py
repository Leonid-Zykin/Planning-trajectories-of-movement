import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy.interpolate import UnivariateSpline

def load_original_data():
    """Загрузить исходные данные карты и пути"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid.npy"))
    path = np.load(os.path.join(data_dir, "path.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal.npy"))
    
    return grid, path, start_goal[0], start_goal[1]

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

def find_bspline_collision_cells(x_bspline, y_bspline, grid, margin=0.3):
    """Найти ячейки, которые пересекает B-сплайн траектория"""
    collision_cells = set()
    
    for i in range(len(x_bspline)):
        x, y = x_bspline[i], y_bspline[i]
        
        # Проверяем точки в окрестности траектории
        for dx in np.arange(-margin, margin + 0.1, 0.1):
            for dy in np.arange(-margin, margin + 0.1, 0.1):
                check_x = int(round(x + dx))
                check_y = int(round(y + dy))
                
                if (0 <= check_x < grid.shape[1] and 0 <= check_y < grid.shape[0]):
                    collision_cells.add((check_y, check_x))  # (row, col)
    
    return collision_cells

def find_path_cells(path):
    """Найти все ячейки, через которые проходит путь A*"""
    path_cells = set()
    for point in path:
        path_cells.add((point[0], point[1]))  # (row, col)
    return path_cells

def modify_map_for_bspline(grid, path, margin=0.3):
    """Модифицировать карту для безопасного прохождения B-сплайн траектории"""
    # Создаем копию карты
    modified_grid = grid.copy()
    
    # Генерируем B-сплайн траекторию
    path_array = np.array(path)
    x_bspline, y_bspline, _ = b_spline_smoothing(path_array)
    
    # Находим ячейки, которые пересекает B-сплайн
    bspline_cells = find_bspline_collision_cells(x_bspline, y_bspline, grid, margin)
    
    # Находим ячейки пути A*
    path_cells = find_path_cells(path)
    
    print(f"B-сплайн пересекает {len(bspline_cells)} ячеек")
    print(f"Путь A* проходит через {len(path_cells)} ячеек")
    
    # Убираем препятствия там, где B-сплайн пересекает их
    for cell in bspline_cells:
        if modified_grid[cell] == 1:  # Если это препятствие
            modified_grid[cell] = 0  # Убираем препятствие
            print(f"Убрано препятствие в ячейке {cell}")
    
    # Добавляем препятствия в места, где робот не проходит
    # Находим ячейки, которые не пересекает ни путь A*, ни B-сплайн
    all_robot_cells = path_cells.union(bspline_cells)
    
    # Добавляем препятствия в свободные углы и края
    size = grid.shape[0]
    new_obstacles = []
    
    # Углы карты (если робот туда не заходит)
    corners = [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]
    for corner in corners:
        if corner not in all_robot_cells and modified_grid[corner] == 0:
            modified_grid[corner] = 1
            new_obstacles.append(corner)
    
    # Края карты
    edges = []
    for i in range(size):
        edges.extend([(0, i), (size-1, i), (i, 0), (i, size-1)])
    
    for edge in edges:
        if edge not in all_robot_cells and modified_grid[edge] == 0:
            # Добавляем препятствие с некоторой вероятностью
            if np.random.random() < 0.3:  # 30% вероятность
                modified_grid[edge] = 1
                new_obstacles.append(edge)
    
    print(f"Добавлено {len(new_obstacles)} новых препятствий")
    
    return modified_grid, x_bspline, y_bspline

def plot_comparison(original_grid, modified_grid, path, x_bspline, y_bspline, 
                   start, goal, filename):
    """Построить сравнение оригинальной и модифицированной карт"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Оригинальная карта
    ax1.set_title('Оригинальная карта')
    plot_map_on_axes(ax1, original_grid, path, start, goal, show_bspline=False)
    
    # Модифицированная карта
    ax2.set_title('Модифицированная карта для B-сплайн')
    plot_map_on_axes(ax2, modified_grid, path, start, goal, 
                    x_bspline, y_bspline, show_bspline=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_map_on_axes(ax, grid, path, start, goal, x_bspline=None, y_bspline=None, show_bspline=False):
    """Построить карту на заданных осях"""
    # Отображаем сетку
    for i in range(grid.shape[0] + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Отображаем препятствия
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1, 
                               facecolor='black', edgecolor='black')
                ax.add_patch(rect)
    
    # Отображаем путь A*
    if path is not None:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=3, label='Путь A*')
        ax.scatter(path_x, path_y, c='red', s=50, zorder=5)
    
    # Отображаем B-сплайн траекторию
    if show_bspline and x_bspline is not None and y_bspline is not None:
        ax.plot(x_bspline, y_bspline, 'orange', linewidth=3, label='B-сплайн траектория')
    
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
    ax.legend()
    ax.grid(True, alpha=0.3)

def main():
    print("Загрузка исходных данных...")
    original_grid, path, start, goal = load_original_data()
    
    print("Модификация карты для B-сплайн траектории...")
    modified_grid, x_bspline, y_bspline = modify_map_for_bspline(original_grid, path)
    
    # Проверяем, что B-сплайн теперь не пересекает препятствия
    bspline_cells = find_bspline_collision_cells(x_bspline, y_bspline, modified_grid)
    collisions = [cell for cell in bspline_cells if modified_grid[cell] == 1]
    
    print(f"Коллизий B-сплайн с препятствиями: {len(collisions)}")
    
    # Сохраняем результаты
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем сравнение
    plot_comparison(original_grid, modified_grid, path, x_bspline, y_bspline,
                   start, goal, os.path.join(output_dir, "map_modification_comparison.png"))
    
    # Сохраняем модифицированную карту
    np.save(os.path.join(output_dir, "grid_modified_for_bspline.npy"), modified_grid)
    np.save(os.path.join(output_dir, "path_modified.npy"), np.array(path))
    np.save(os.path.join(output_dir, "start_goal_modified.npy"), np.array([start, goal]))
    
    print(f"Результаты сохранены в {output_dir}")
    print("Модификация карты завершена!")

if __name__ == "__main__":
    main()
