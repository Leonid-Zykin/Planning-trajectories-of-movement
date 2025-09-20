import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import heapq
from typing import List, Tuple, Optional
import os

class AStar:
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.rows, self.cols = grid.shape
        
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Эвристическая функция (манхэттенское расстояние)"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Получить соседние ячейки (8-связность)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = pos[0] + dx, pos[1] + dy
                if (0 <= new_x < self.rows and 0 <= new_y < self.cols and 
                    self.grid[new_x, new_y] == 0):  # 0 = свободная ячейка
                    neighbors.append((new_x, new_y))
        return neighbors
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Найти путь от start до goal используя алгоритм A*"""
        if self.grid[start[0], start[1]] == 1 or self.grid[goal[0], goal[1]] == 1:
            return None
            
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Восстановить путь
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

def create_binary_map(size: int = 10, obstacle_ratio: float = 0.35) -> np.ndarray:
    """Создать бинарную карту с заданным соотношением препятствий"""
    grid = np.zeros((size, size), dtype=int)
    
    # Создаем препятствия случайным образом
    total_cells = size * size
    num_obstacles = int(total_cells * obstacle_ratio)
    
    # Создаем список всех позиций и перемешиваем
    positions = [(i, j) for i in range(size) for j in range(size)]
    np.random.seed(42)  # Для воспроизводимости
    np.random.shuffle(positions)
    
    # Размещаем препятствия
    for i in range(num_obstacles):
        grid[positions[i]] = 1
    
    return grid

def find_good_start_goal(grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Найти подходящие начальную и конечную точки"""
    size = grid.shape[0]
    
    # Пробуем разные комбинации точек
    for start in [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]:
        for goal in [(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)]:
            if start != goal and grid[start] == 0 and grid[goal] == 0:
                astar = AStar(grid)
                path = astar.find_path(start, goal)
                if path and len(path) >= 10:
                    # Проверяем количество поворотов
                    turns = count_turns(path)
                    if turns >= 3:
                        return start, goal
    
    # Если не нашли подходящие угловые точки, пробуем другие
    for _ in range(100):
        start = (np.random.randint(0, size), np.random.randint(0, size))
        goal = (np.random.randint(0, size), np.random.randint(0, size))
        if start != goal and grid[start] == 0 and grid[goal] == 0:
            astar = AStar(grid)
            path = astar.find_path(start, goal)
            if path and len(path) >= 10:
                turns = count_turns(path)
                if turns >= 3:
                    return start, goal
    
    return (0, 0), (size-1, size-1)  # Fallback

def count_turns(path: List[Tuple[int, int]]) -> int:
    """Подсчитать количество поворотов в пути"""
    if len(path) < 3:
        return 0
    
    turns = 0
    for i in range(1, len(path) - 1):
        # Векторы направления
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        
        # Если векторы не коллинеарны, это поворот
        if v1 != v2:
            turns += 1
    
    return turns

def plot_map_and_path(grid: np.ndarray, path: List[Tuple[int, int]], 
                     start: Tuple[int, int], goal: Tuple[int, int], 
                     filename: str):
    """Построить карту с путем"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
    
    # Отображаем путь
    if path:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        ax.plot(path_x, path_y, 'r-', linewidth=3, label='Путь A*')
        ax.scatter(path_x, path_y, c='red', s=50, zorder=5)
    
    # Отмечаем начальную и конечную точки
    ax.scatter(start[1], start[0], c='green', s=200, marker='s', 
               label='Начальная точка', zorder=6)
    ax.scatter(goal[1], goal[0], c='blue', s=200, marker='*', 
               label='Конечная точка', zorder=6)
    
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Инвертируем Y для правильного отображения
    ax.set_xlabel('X (столбцы)')
    ax.set_ylabel('Y (строки)')
    ax.set_title('Бинарная карта и путь, найденный алгоритмом A*')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Создаем бинарную карту
    print("Создание бинарной карты 10x10...")
    grid = create_binary_map(10, 0.35)
    
    # Находим подходящие начальную и конечную точки
    print("Поиск подходящих начальной и конечной точек...")
    start, goal = find_good_start_goal(grid)
    
    # Применяем алгоритм A*
    print("Применение алгоритма A*...")
    astar = AStar(grid)
    path = astar.find_path(start, goal)
    
    if path is None:
        print("Путь не найден!")
        return
    
    print(f"Найден путь длиной {len(path)} ячеек")
    print(f"Количество поворотов: {count_turns(path)}")
    print(f"Начальная точка: {start}")
    print(f"Конечная точка: {goal}")
    
    # Сохраняем карту и путь
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_map_and_path(grid, path, start, goal, 
                     os.path.join(output_dir, "astar_path.png"))
    
    # Сохраняем данные для дальнейшего использования
    np.save(os.path.join(output_dir, "grid.npy"), grid)
    np.save(os.path.join(output_dir, "path.npy"), np.array(path))
    np.save(os.path.join(output_dir, "start_goal.npy"), np.array([start, goal]))
    
    print(f"Результаты сохранены в {output_dir}")

if __name__ == "__main__":
    main()
