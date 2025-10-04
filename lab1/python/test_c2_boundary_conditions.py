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

def generate_c2_trajectory_natural(path_points):
    """Генерация C²-гладкой траектории с естественными граничными условиями"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='natural')
    y_spline = CubicSpline(t, y_coords, bc_type='natural')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c2_trajectory_periodic(path_points):
    """Генерация C²-гладкой траектории с периодическими граничными условиями"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='periodic')
    y_spline = CubicSpline(t, y_coords, bc_type='periodic')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c2_trajectory_not_a_knot(path_points):
    """Генерация C²-гладкой траектории с not-a-knot граничными условиями"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='not-a-knot')
    y_spline = CubicSpline(t, y_coords, bc_type='not-a-knot')
    
    t_dense = np.linspace(0, 1, 1000)
    x_traj = x_spline(t_dense)
    y_traj = y_spline(t_dense)
    
    return x_traj, y_traj, t_dense

def generate_c2_trajectory_clamped(path_points):
    """Генерация C²-гладкой траектории с закрепленными граничными условиями"""
    x_coords = path_points[:, 1]
    y_coords = path_points[:, 0]
    t = np.linspace(0, 1, len(path_points))
    
    x_spline = CubicSpline(t, x_coords, bc_type='clamped')
    y_spline = CubicSpline(t, y_coords, bc_type='clamped')
    
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

def plot_curvature_comparison_c2_methods(t_natural, curvature_natural, t_periodic, curvature_periodic,
                                       t_notaknot, curvature_notaknot, t_clamped, curvature_clamped, filename):
    """Построение сравнения кривизн C²-траекторий с разными граничными условиями"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(t_natural, curvature_natural, 'g-', linewidth=2, label='C²-гладкая (natural)')
    ax.plot(t_periodic, curvature_periodic, 'b-', linewidth=2, label='C²-гладкая (periodic)')
    ax.plot(t_notaknot, curvature_notaknot, 'r-', linewidth=2, label='C²-гладкая (not-a-knot)')
    ax.plot(t_clamped, curvature_clamped, 'm-', linewidth=2, label='C²-гладкая (clamped)')
    
    ax.set_xlabel('Параметр t')
    ax.set_ylabel('Кривизна κ')
    ax.set_title('Сравнение кривизн C²-траекторий с разными граничными условиями')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка модифицированных данных...")
    grid, path, start, goal = load_modified_data()
    
    print("Тестирование разных граничных условий для C²-траектории...")
    
    path_array = np.array(path)
    
    # Генерируем C²-траектории с разными граничными условиями
    x_natural, y_natural, t_natural = generate_c2_trajectory_natural(path_array)
    x_periodic, y_periodic, t_periodic = generate_c2_trajectory_periodic(path_array)
    x_notaknot, y_notaknot, t_notaknot = generate_c2_trajectory_not_a_knot(path_array)
    x_clamped, y_clamped, t_clamped = generate_c2_trajectory_clamped(path_array)
    
    # Вычисляем кривизны
    curvature_natural = calculate_curvature(x_natural, y_natural, t_natural)
    curvature_periodic = calculate_curvature(x_periodic, y_periodic, t_periodic)
    curvature_notaknot = calculate_curvature(x_notaknot, y_notaknot, t_notaknot)
    curvature_clamped = calculate_curvature(x_clamped, y_clamped, t_clamped)
    
    # Создаем папку для результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_curvature_comparison_c2_methods(t_natural, curvature_natural, t_periodic, curvature_periodic,
                                       t_notaknot, curvature_notaknot, t_clamped, curvature_clamped,
                                       os.path.join(output_dir, "c2_boundary_conditions_comparison.png"))
    
    print(f"График сохранен в {output_dir}")
    
    # Выводим статистику
    print("\nСтатистика кривизн C²-траекторий:")
    print(f"Natural: средняя = {np.mean(curvature_natural):.4f}, макс = {np.max(curvature_natural):.4f}")
    print(f"Periodic: средняя = {np.mean(curvature_periodic):.4f}, макс = {np.max(curvature_periodic):.4f}")
    print(f"Not-a-knot: средняя = {np.mean(curvature_notaknot):.4f}, макс = {np.max(curvature_notaknot):.4f}")
    print(f"Clamped: средняя = {np.mean(curvature_clamped):.4f}, макс = {np.max(curvature_clamped):.4f}")

if __name__ == "__main__":
    main()
