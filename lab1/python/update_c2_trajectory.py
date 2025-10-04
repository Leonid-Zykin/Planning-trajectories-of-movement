import numpy as np
import matplotlib.pyplot as plt
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
    """Генерация C²-гладкой траектории с улучшенными граничными условиями"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    # Используем natural граничные условия вместо clamped для лучшей кривизны на концах
    x_spline = CubicSpline(t, x_coords, bc_type='natural')
    y_spline = CubicSpline(t, y_coords, bc_type='natural')
    
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
    ax.set_title('Сравнение кривизн траекторий (улучшенная C²-траектория)')
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
    ax.set_title('Сравнение кривизн траекторий (без C⁰-гладкой, улучшенная C²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем разумные пределы по Y для лучшей видимости
    max_curvature = max(np.max(curvature_c1), np.max(curvature_c2), np.max(curvature_bspline))
    ax.set_ylim(0, max_curvature * 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка модифицированных данных...")
    grid, path, start, goal = load_modified_data()
    
    print("Генерация улучшенных траекторий...")
    path_array = np.array(path)
    
    x_c0, y_c0, t_c0 = generate_c0_trajectory(path_array)
    x_c1, y_c1, t_c1 = generate_c1_trajectory(path_array)
    x_c2, y_c2, t_c2 = generate_c2_trajectory(path_array)
    x_bspline, y_bspline, t_bspline = b_spline_smoothing(path_array)
    
    print("Вычисление кривизн...")
    curvature_c0 = calculate_curvature(x_c0, y_c0, t_c0)
    curvature_c1 = calculate_curvature(x_c1, y_c1, t_c1)
    curvature_c2 = calculate_curvature(x_c2, y_c2, t_c2)
    curvature_bspline = calculate_curvature(x_bspline, y_bspline, t_bspline)
    
    # Создаем папку для результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Создание обновленных графиков...")
    plot_curvature_comparison(t_c0, curvature_c0, t_c1, curvature_c1,
                             t_c2, curvature_c2, t_bspline, curvature_bspline,
                             os.path.join(output_dir, "curvature_comparison_improved.png"))
    
    plot_curvature_comparison_without_c0(t_c1, curvature_c1, t_c2, curvature_c2,
                                       t_bspline, curvature_bspline,
                                       os.path.join(output_dir, "curvature_comparison_without_c0_improved.png"))
    
    # Сохраняем обновленные данные траекторий
    np.savez(os.path.join(output_dir, "trajectories_improved.npz"),
             x_c0=x_c0, y_c0=y_c0, t_c0=t_c0,
             x_c1=x_c1, y_c1=y_c1, t_c1=t_c1,
             x_c2=x_c2, y_c2=y_c2, t_c2=t_c2,
             x_bspline=x_bspline, y_bspline=y_bspline, t_bspline=t_bspline,
             curvature_c0=curvature_c0, curvature_c1=curvature_c1,
             curvature_c2=curvature_c2, curvature_bspline=curvature_bspline)
    
    print(f"Результаты сохранены в {output_dir}")
    
    # Выводим статистику
    print("\nСтатистика кривизн (улучшенная C²-траектория):")
    print(f"C⁰-гладкая: средняя = {np.mean(curvature_c0):.4f}, макс = {np.max(curvature_c0):.4f}")
    print(f"C¹-гладкая: средняя = {np.mean(curvature_c1):.4f}, макс = {np.max(curvature_c1):.4f}")
    print(f"C²-гладкая: средняя = {np.mean(curvature_c2):.4f}, макс = {np.max(curvature_c2):.4f}")
    print(f"B-сплайн: средняя = {np.mean(curvature_bspline):.4f}, макс = {np.max(curvature_bspline):.4f}")

if __name__ == "__main__":
    main()
