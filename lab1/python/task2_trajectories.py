import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize
import os

def load_path_data():
    """Загрузить данные пути из предыдущего задания"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task1"
    grid = np.load(os.path.join(data_dir, "grid.npy"))
    path = np.load(os.path.join(data_dir, "path.npy"))
    start_goal = np.load(os.path.join(data_dir, "start_goal.npy"))
    
    return grid, path, start_goal[0], start_goal[1]

def generate_c0_trajectory(path_points):
    """Генерация C⁰-гладкой траектории (кусочно-линейная)"""
    # C⁰ означает непрерывность функции (без разрывов)
    # Просто соединяем точки прямыми линиями
    x_coords = path_points[:, 1]  # столбцы как x
    y_coords = path_points[:, 0]  # строки как y
    
    # Создаем параметр t для интерполяции
    t = np.linspace(0, 1, len(path_points))
    
    # Линейная интерполяция
    x_interp = interp1d(t, x_coords, kind='linear')
    y_interp = interp1d(t, y_coords, kind='linear')
    
    # Создаем более плотную сетку для плавной траектории
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_interp(t_dense)
    y_traj = y_interp(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c1_trajectory(path_points):
    """Генерация C¹-гладкой траектории (непрерывная первая производная)"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    # Кубическая сплайн-интерполяция для C¹ гладкости
    x_spline = CubicSpline(t, x_coords, bc_type='natural')
    y_spline = CubicSpline(t, y_coords, bc_type='natural')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c2_trajectory(path_points):
    """Генерация C²-гладкой траектории (непрерывная вторая производная)"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    # Используем кубический сплайн с условиями на вторые производные
    x_spline = CubicSpline(t, x_coords, bc_type='clamped')
    y_spline = CubicSpline(t, y_coords, bc_type='clamped')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def b_spline_smoothing(path_points, degree=3):
    """Сглаживание траектории с помощью B-сплайна"""
    from scipy.interpolate import BSpline
    
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    
    # Создаем узлы для B-сплайна
    n = len(path_points)
    t = np.linspace(0, 1, n)
    
    # Добавляем дополнительные узлы на концах для граничных условий
    knots = np.concatenate([[0] * degree, t, [1] * degree])
    
    # Создаем B-сплайны
    x_bspline = BSpline(knots, x_coords, degree)
    y_bspline = BSpline(knots, y_coords, degree)
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_bspline(t_dense)
    y_traj = y_bspline(t_dense)
    
    return x_traj, y_traj, t_dense

def calculate_curvature(x, y, t):
    """Вычисление кривизны траектории"""
    # Первые производные
    dx_dt = np.gradient(x, t)
    dy_dt = np.gradient(y, t)
    
    # Вторые производные
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)
    
    # Кривизна: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2)**(3/2)
    
    # Избегаем деления на ноль
    curvature = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    
    return curvature

def plot_trajectories(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline, 
                     path_points, filename):
    """Построение всех траекторий на одном графике"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Отображаем исходные точки пути
    ax.scatter(path_points[:, 1], path_points[:, 0], c='red', s=100, 
               label='Точки пути A*', zorder=5)
    
    # Отображаем траектории
    ax.plot(x_c0, y_c0, 'b-', linewidth=2, label='C⁰-гладкая (линейная)')
    ax.plot(x_c1, y_c1, 'g-', linewidth=2, label='C¹-гладкая (кубический сплайн)')
    ax.plot(x_c2, y_c2, 'm-', linewidth=2, label='C²-гладкая (кубический сплайн)')
    ax.plot(x_bspline, y_bspline, 'orange', linewidth=2, label='B-сплайн сглаживание')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Сравнение траекторий с разной гладкостью')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
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

def main():
    print("Загрузка данных пути...")
    grid, path, start, goal = load_path_data()
    
    print("Генерация C⁰-гладкой траектории...")
    x_c0, y_c0, t_c0 = generate_c0_trajectory(path)
    
    print("Генерация C¹-гладкой траектории...")
    x_c1, y_c1, t_c1 = generate_c1_trajectory(path)
    
    print("Генерация C²-гладкой траектории...")
    x_c2, y_c2, t_c2 = generate_c2_trajectory(path)
    
    print("B-сплайн сглаживание...")
    x_bspline, y_bspline, t_bspline = b_spline_smoothing(path)
    
    print("Вычисление кривизн...")
    curvature_c0 = calculate_curvature(x_c0, y_c0, t_c0)
    curvature_c1 = calculate_curvature(x_c1, y_c1, t_c1)
    curvature_c2 = calculate_curvature(x_c2, y_c2, t_c2)
    curvature_bspline = calculate_curvature(x_bspline, y_bspline, t_bspline)
    
    # Создаем папку для результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Создание графиков...")
    plot_trajectories(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, x_bspline, y_bspline,
                     path, os.path.join(output_dir, "trajectories_comparison.png"))
    
    plot_curvature_comparison(t_c0, curvature_c0, t_c1, curvature_c1,
                             t_c2, curvature_c2, t_bspline, curvature_bspline,
                             os.path.join(output_dir, "curvature_comparison.png"))
    
    # Сохраняем данные траекторий
    np.savez(os.path.join(output_dir, "trajectories.npz"),
             x_c0=x_c0, y_c0=y_c0, t_c0=t_c0,
             x_c1=x_c1, y_c1=y_c1, t_c1=t_c1,
             x_c2=x_c2, y_c2=y_c2, t_c2=t_c2,
             x_bspline=x_bspline, y_bspline=y_bspline, t_bspline=t_bspline,
             curvature_c0=curvature_c0, curvature_c1=curvature_c1,
             curvature_c2=curvature_c2, curvature_bspline=curvature_bspline)
    
    print(f"Результаты сохранены в {output_dir}")
    
    # Выводим статистику
    print("\nСтатистика кривизн:")
    print(f"C⁰-гладкая: средняя = {np.mean(curvature_c0):.4f}, макс = {np.max(curvature_c0):.4f}")
    print(f"C¹-гладкая: средняя = {np.mean(curvature_c1):.4f}, макс = {np.max(curvature_c1):.4f}")
    print(f"C²-гладкая: средняя = {np.mean(curvature_c2):.4f}, макс = {np.max(curvature_c2):.4f}")
    print(f"B-сплайн: средняя = {np.mean(curvature_bspline):.4f}, макс = {np.max(curvature_bspline):.4f}")

if __name__ == "__main__":
    main()
