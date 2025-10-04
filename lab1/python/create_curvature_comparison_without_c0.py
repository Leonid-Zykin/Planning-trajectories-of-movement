import numpy as np
import matplotlib.pyplot as plt
import os

def load_trajectory_data():
    """Загрузить данные траекторий"""
    data_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    data = np.load(os.path.join(data_dir, "trajectories_modified.npz"))
    
    return (data['t_c0'], data['curvature_c0'],
            data['t_c1'], data['curvature_c1'],
            data['t_c2'], data['curvature_c2'],
            data['t_bspline'], data['curvature_bspline'])

def plot_curvature_comparison_without_c0(t_c1, curvature_c1, t_c2, curvature_c2, 
                                       t_bspline, curvature_bspline, filename):
    """Построение сравнения кривизн без C⁰-гладкой траектории"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Отображаем только C¹, C² и B-сплайн
    ax.plot(t_c1, curvature_c1, 'g-', linewidth=2, label='C¹-гладкая')
    ax.plot(t_c2, curvature_c2, 'm-', linewidth=2, label='C²-гладкая')
    ax.plot(t_bspline, curvature_bspline, 'orange', linewidth=2, label='B-сплайн')
    
    ax.set_xlabel('Параметр t')
    ax.set_ylabel('Кривизна κ')
    ax.set_title('Сравнение кривизн траекторий (без C⁰-гладкой)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Устанавливаем разумные пределы по Y для лучшей видимости
    max_curvature = max(np.max(curvature_c1), np.max(curvature_c2), np.max(curvature_bspline))
    ax.set_ylim(0, max_curvature * 1.1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("Загрузка данных траекторий...")
    t_c0, curvature_c0, t_c1, curvature_c1, t_c2, curvature_c2, t_bspline, curvature_bspline = load_trajectory_data()
    
    print("Создание графика сравнения кривизн без C⁰-гладкой траектории...")
    
    # Создаем папку для результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab1/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_curvature_comparison_without_c0(t_c1, curvature_c1, t_c2, curvature_c2,
                                       t_bspline, curvature_bspline,
                                       os.path.join(output_dir, "curvature_comparison_without_c0.png"))
    
    print(f"График сохранен в {output_dir}")
    
    # Выводим статистику для сравнения
    print("\nСтатистика кривизн (без C⁰):")
    print(f"C¹-гладкая: средняя = {np.mean(curvature_c1):.4f}, макс = {np.max(curvature_c1):.4f}")
    print(f"C²-гладкая: средняя = {np.mean(curvature_c2):.4f}, макс = {np.max(curvature_c2):.4f}")
    print(f"B-сплайн: средняя = {np.mean(curvature_bspline):.4f}, макс = {np.max(curvature_bspline):.4f}")

if __name__ == "__main__":
    main()
