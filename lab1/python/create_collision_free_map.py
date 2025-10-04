import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
import heapq
from typing import List, Tuple, Set

def create_collision_free_map(size: int = 10, obstacle_ratio: float = 0.35, seed: int = 42) -> np.ndarray:
    """Создание карты без коллизий для траекторий"""
    np.random.seed(seed)
    grid = np.zeros((size, size), dtype=int)
    
    # Создаем путь сначала, чтобы избежать коллизий
    # Выбираем начальную и конечную точки на противоположных сторонах
    start = (0, 2)
    goal = (9, 7)
    
    # Создаем путь с несколькими поворотами
    path_points = [
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2),
        (5, 3), (5, 4), (5, 5), (6, 5), (7, 5), (8, 5),
        (8, 6), (8, 7), (9, 7)
    ]
    
    # Размещаем препятствия, избегая области вокруг пути
    total_cells = size * size
    num_obstacles = int(total_cells * obstacle_ratio)
    
    # Создаем маску для области вокруг пути
    path_mask = np.zeros((size, size), dtype=bool)
    for i, j in path_points:
        # Расширяем область вокруг пути
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    path_mask[ni, nj] = True
    
    # Получаем доступные позиции для препятствий
    available_positions = []
    for i in range(size):
        for j in range(size):
            if not path_mask[i, j]:
                available_positions.append((i, j))
    
    # Размещаем препятствия в доступных позициях
    if len(available_positions) >= num_obstacles:
        obstacle_positions = np.random.choice(len(available_positions), num_obstacles, replace=False)
        for pos_idx in obstacle_positions:
            i, j = available_positions[pos_idx]
            grid[i, j] = 1
    
    return grid, path_points, start, goal

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Эвристическая функция (манхэттенское расстояние)"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Алгоритм A* для поиска пути"""
    rows, cols = grid.shape
    
    # Проверяем валидность начальной и конечной точек
    if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
        return []
    
    # Инициализация
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    closed_set = set()
    
    # Возможные направления движения (8-связность)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current in closed_set:
            continue
            
        closed_set.add(current)
        
        if current == goal:
            # Восстанавливаем путь
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Проверяем соседние ячейки
        for di, dj in directions:
            neighbor = (current[0] + di, current[1] + dj)
            
            # Проверяем границы
            if (neighbor[0] < 0 or neighbor[0] >= rows or 
                neighbor[1] < 0 or neighbor[1] >= cols):
                continue
            
            # Проверяем препятствие
            if grid[neighbor[0], neighbor[1]] == 1:
                continue
            
            if neighbor in closed_set:
                continue
            
            # Вычисляем стоимость движения
            if abs(di) + abs(dj) == 2:  # Диагональное движение
                tentative_g = g_score[current] + 1.414
            else:  # Ортогональное движение
                tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

def count_turns(path: List[Tuple[int, int]]) -> int:
    """Подсчет количества поворотов в пути"""
    if len(path) < 3:
        return 0
    
    turns = 0
    for i in range(1, len(path) - 1):
        # Вычисляем направления
        prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        
        # Проверяем изменение направления
        if prev_dir != next_dir:
            turns += 1
    
    return turns

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

def calculate_curvature(x, y, t):
    """Вычисление кривизны траектории"""
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)
    
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return curvature

def check_collisions(grid: np.ndarray, x_traj: np.ndarray, y_traj: np.ndarray) -> Tuple[bool, int]:
    """Проверка коллизий траектории с препятствиями"""
    collisions = 0
    
    for i in range(len(x_traj)):
        x, y = x_traj[i], y_traj[i]
        
        # Проверяем, попадает ли точка в препятствие
        if (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and 
            grid[int(y), int(x)] == 1):
            collisions += 1
    
    return collisions > 0, collisions

def plot_map_with_trajectories(grid: np.ndarray, path: List[Tuple[int, int]], 
                              start: Tuple[int, int], goal: Tuple[int, int],
                              x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline,
                              filename: str):
    """Построение карты со всеми траекториями"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем карту
    ax.imshow(grid, cmap='binary', origin='lower')
    
    # Отображаем траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=2, label='C⁰-гладкая')
    ax.plot(x_c1, y_c1, 'g-', linewidth=2, label='C¹-гладкая')
    ax.plot(x_c2, y_c2, 'm-', linewidth=2, label='C²-гладкая')
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=2, label='B-сплайн')
    
    # Отображаем путь A*
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 1], path_array[:, 0], 'r--', linewidth=2, alpha=0.7, label='Путь A*')
        ax.scatter(path_array[:, 1], path_array[:, 0], c='red', s=30, zorder=5)
    
    # Отображаем начальную и конечную точки
    ax.scatter(start[1], start[0], c='green', s=200, marker='s', label='Начало', zorder=10)
    ax.scatter(goal[1], goal[0], c='blue', s=200, marker='*', label='Конец', zorder=10)
    
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Карта с траекториями (без коллизий)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectories_comparison(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, 
                               x_bspline, y_bspline, path_points, filename):
    """Построение сравнения траекторий"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=2, label='C⁰-гладкая')
    ax.plot(x_c1, y_c1, 'g-', linewidth=2, label='C¹-гладкая')
    ax.plot(x_c2, y_c2, 'm-', linewidth=2, label='C²-гладкая')
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=2, label='B-сплайн')
    
    # Отображаем точки пути A*
    ax.scatter(path_points[:, 1], path_points[:, 0], c='red', s=50, 
               label='Точки пути A*', zorder=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Сравнение траекторий с разной гладкостью')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curvature_comparison(t_c0, curvature_c0, t_c1, curvature_c1, 
                             t_c2, curvature_c2, t_bspline, curvature_bspline, filename):
    """Построение сравнения кривизн"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(t_c0, curvature_c0, 'b-', linewidth=2, label='C⁰-гладкая')
    ax.plot(t_c1, curvature_c1, 'g-', linewidth=2, label='C¹-гладкая')
    ax.plot(t_c2, curvature_c2, 'm-', linewidth=2, label='C²-гладкая')
    ax.plot(t_bspline, curvature_bspline, 'orange', linewidth=2, label='B-сплайн')
    
    ax.set_xlabel('Параметр t')
    ax.set_ylabel('Кривизна κ')
    ax.set_title('Сравнение кривизн траекторий')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_curvature_comparison_without_c0(t_c1, curvature_c1, t_c2, curvature_c2, 
                                        t_bspline, curvature_bspline, filename):
    """Построение сравнения кривизн без C⁰-гладкой траектории"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(t_c1, curvature_c1, 'g-', linewidth=2, label='C¹-гладкая')
    ax.plot(t_c2, curvature_c2, 'm-', linewidth=2, label='C²-гладкая')
    ax.plot(t_bspline, curvature_bspline, 'orange', linewidth=2, label='B-сплайн')
    
    ax.set_xlabel('Параметр t')
    ax.set_ylabel('Кривизна κ')
    ax.set_title('Сравнение кривизн траекторий (без C⁰-гладкой)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_multiple_bsplines(path_points, smoothing_factors, filename):
    """Построение нескольких B-сплайн траекторий с разными параметрами"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['orange', 'red', 'purple', 'brown', 'pink']
    
    for i, s in enumerate(smoothing_factors):
        x_bspline, y_bspline, _ = generate_bspline_trajectory(path_points, s)
        label = f'B-сплайн (s={s})' if s is not None else 'B-сплайн (авто)'
        ax.plot(x_bspline, y_bspline, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    
    # Отображаем точки пути A*
    ax.scatter(path_points[:, 1], path_points[:, 0], c='red', s=50, 
               label='Точки пути A*', zorder=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('B-сплайн траектории с разными параметрами сглаживания')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Создание карты без коллизий...")
    
    # Создаем карту без коллизий
    grid, path_points, start, goal = create_collision_free_map(size=10, obstacle_ratio=0.35, seed=42)
    
    print(f"Создан путь длиной {len(path_points)} ячеек с {count_turns(path_points)} поворотами")
    print(f"Начальная точка: {start}, Конечная точка: {goal}")
    
    # Создаем папки для результатов
    os.makedirs("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1", exist_ok=True)
    os.makedirs("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2", exist_ok=True)
    
    # Сохраняем данные
    np.save("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1/grid_final.npy", grid)
    np.save("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1/path_final.npy", path_points)
    np.save("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1/start_goal_final.npy", 
            np.array([start, goal]))
    
    print("Генерация траекторий...")
    path_array = np.array(path_points)
    
    # Генерируем траектории
    x_c0, y_c0, t_c0 = generate_c0_trajectory(path_array)
    x_c1, y_c1, t_c1 = generate_c1_trajectory(path_array)
    x_c2, y_c2, t_c2 = generate_c2_trajectory(path_array)
    x_bspline, y_bspline, t_bspline = generate_bspline_trajectory(path_array)
    
    # Проверяем коллизии
    print("Проверка коллизий...")
    c0_collision, c0_count = check_collisions(grid, x_c0, y_c0)
    c1_collision, c1_count = check_collisions(grid, x_c1, y_c1)
    c2_collision, c2_count = check_collisions(grid, x_c2, y_c2)
    bspline_collision, bspline_count = check_collisions(grid, x_bspline, y_bspline)
    
    print(f"C⁰-гладкая: коллизии = {c0_collision}, количество = {c0_count}")
    print(f"C¹-гладкая: коллизии = {c1_collision}, количество = {c1_count}")
    print(f"C²-гладкая: коллизии = {c2_collision}, количество = {c2_count}")
    print(f"B-сплайн: коллизии = {bspline_collision}, количество = {bspline_count}")
    
    # Вычисляем кривизны
    print("Вычисление кривизн...")
    curvature_c0 = calculate_curvature(x_c0, y_c0, t_c0)
    curvature_c1 = calculate_curvature(x_c1, y_c1, t_c1)
    curvature_c2 = calculate_curvature(x_c2, y_c2, t_c2)
    curvature_bspline = calculate_curvature(x_bspline, y_bspline, t_bspline)
    
    # Строим графики
    print("Создание графиков...")
    
    # Карта со всеми траекториями
    plot_map_with_trajectories(grid, path_points, start, goal,
                              x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline,
                              "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1/astar_path_with_trajectories_final.png")
    
    plot_trajectories_comparison(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, 
                               x_bspline, y_bspline, path_array,
                               "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/trajectories_comparison_final.png")
    
    plot_curvature_comparison(t_c0, curvature_c0, t_c1, curvature_c1,
                             t_c2, curvature_c2, t_bspline, curvature_bspline,
                             "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/curvature_comparison_final.png")
    
    plot_curvature_comparison_without_c0(t_c1, curvature_c1, t_c2, curvature_c2,
                                       t_bspline, curvature_bspline,
                                       "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/curvature_comparison_without_c0_final.png")
    
    # Строим несколько B-сплайн траекторий с разными параметрами
    smoothing_factors = [None, 0.1, 0.5, 1.0, 2.0]
    plot_multiple_bsplines(path_array, smoothing_factors,
                          "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/multiple_bsplines_final.png")
    
    # Сохраняем данные траекторий
    np.savez("/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2/trajectories_final.npz",
             x_c0=x_c0, y_c0=y_c0, t_c0=t_c0,
             x_c1=x_c1, y_c1=y_c1, t_c1=t_c1,
             x_c2=x_c2, y_c2=y_c2, t_c2=t_c2,
             x_bspline=x_bspline, y_bspline=y_bspline, t_bspline=t_bspline,
             curvature_c0=curvature_c0, curvature_c1=curvature_c1,
             curvature_c2=curvature_c2, curvature_bspline=curvature_bspline)
    
    print("Результаты сохранены!")
    
    # Выводим статистику
    print("\nСтатистика кривизн:")
    print(f"C⁰-гладкая: средняя = {np.mean(curvature_c0):.4f}, макс = {np.max(curvature_c0):.4f}")
    print(f"C¹-гладкая: средняя = {np.mean(curvature_c1):.4f}, макс = {np.max(curvature_c1):.4f}")
    print(f"C²-гладкая: средняя = {np.mean(curvature_c2):.4f}, макс = {np.max(curvature_c2):.4f}")
    print(f"B-сплайн: средняя = {np.mean(curvature_bspline):.4f}, макс = {np.max(curvature_bspline):.4f}")

if __name__ == "__main__":
    main()
