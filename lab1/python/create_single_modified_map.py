import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def load_modified_data():
    """Загрузить модифицированные данные"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid_modified_for_bspline.npy"))
    path = np.load(os.path.join(data_dir, "path_modified.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal_modified.npy"))
    
    return grid, path, start_goal[0], start_goal[1]

def plot_single_map(grid, path, start, goal, filename):
    """Построить только модифицированную карту"""
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
    
    # Отображаем путь A*
    if path is not None:
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
    ax.invert_yaxis()
    ax.set_xlabel('X (столбцы)')
    ax.set_ylabel('Y (строки)')
    ax.set_title('Бинарная карта и путь, найденный алгоритмом A*')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка модифицированных данных...")
    grid, path, start, goal = load_modified_data()
    
    print("Создание изображения модифицированной карты...")
    
    # Сохраняем только модифицированную карту
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_single_map(grid, path, start, goal, 
                   os.path.join(output_dir, "astar_path_modified_single.png"))
    
    print(f"Изображение сохранено в {output_dir}")
    print("Создана только модифицированная карта!")

if __name__ == "__main__":
    main()
