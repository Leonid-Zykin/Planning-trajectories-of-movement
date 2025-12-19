"""
Симуляция свободного движения Segway
"""

import numpy as np
import matplotlib.pyplot as plt
from segway_model import SegwayModel
import os

def main():
    # Создание модели
    # Параметры подобраны для более явного демонстрирования неустойчивости
    # Уменьшен момент инерции платформы для большей неустойчивости
    segway = SegwayModel(M=10.0, m=2.0, L=0.5, R=0.1, I_p=0.6, I_w=0.1, g=9.81)
    
    # Начальные условия: отклонение от вертикали
    # Увеличено для более явного демонстрирования неустойчивости
    theta0 = np.deg2rad(20.0)  # 20 градусов от вертикали
    theta_dot0 = np.deg2rad(5.0)  # Начальная угловая скорость для ускорения падения
    phi0 = 0.0
    phi_dot0 = 0.0
    x0 = 0.0
    x_dot0 = 0.0
    
    initial_state = [theta0, theta_dot0, phi0, phi_dot0, x0, x_dot0]
    
    # Временной интервал увеличен для демонстрации падения
    t_span = np.linspace(0, 25.0, 25000)
    
    # Симуляция свободного движения
    print("Симуляция свободного движения Segway...")
    sol = segway.simulate_free(initial_state, t_span)
    
    theta = sol[:, 0]
    theta_dot = sol[:, 1]
    phi = sol[:, 2]
    phi_dot = sol[:, 3]
    x = sol[:, 4]
    x_dot = sol[:, 5]
    
    # Построение упрощенного графика - только основные переменные
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Угол наклона платформы
    theta_deg = np.rad2deg(theta)
    axes[0].plot(t_span, theta_deg, 'b-', linewidth=2.5, label='Угол наклона')
    axes[0].set_xlabel('Время, с', fontsize=11)
    axes[0].set_ylabel('Угол наклона θ, град', fontsize=11)
    axes[0].set_title('Угол наклона платформы (свободное движение)\nНеустойчивость: амплитуда растет', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Вертикальное положение')
    # Добавляем линии, показывающие критический угол падения
    axes[0].axhline(y=45, color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Критический угол (45°)')
    axes[0].axhline(y=-45, color='orange', linestyle=':', alpha=0.7, linewidth=2)
    
    # Вычисляем огибающую для демонстрации экспоненциального роста
    # Находим локальные максимумы и минимумы
    from scipy.signal import find_peaks
    peaks_pos, _ = find_peaks(theta_deg, distance=100)
    peaks_neg, _ = find_peaks(-theta_deg, distance=100)
    
    if len(peaks_pos) > 0 and len(peaks_neg) > 0:
        # Рисуем огибающую для демонстрации роста амплитуды
        envelope_upper = np.interp(t_span, t_span[peaks_pos], theta_deg[peaks_pos])
        envelope_lower = np.interp(t_span, t_span[peaks_neg], -theta_deg[peaks_neg])
        axes[0].plot(t_span, envelope_upper, 'r--', linewidth=2, alpha=0.7, label='Рост амплитуды')
        axes[0].plot(t_span, -envelope_lower, 'r--', linewidth=2, alpha=0.7)
    
    # Улучшаем масштаб для лучшей визуализации роста
    max_theta = np.max(np.abs(theta_deg))
    axes[0].set_ylim([-max(50, max_theta*1.3), max(50, max_theta*1.3)])
    axes[0].legend(loc='upper left', fontsize=9)
    
    # Угол поворота колес
    axes[1].plot(t_span, np.rad2deg(phi), 'g-', linewidth=2.5)
    axes[1].set_xlabel('Время, с', fontsize=11)
    axes[1].set_ylabel('Угол поворота φ, град', fontsize=11)
    axes[1].set_title('Угол поворота колес', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Горизонтальное положение
    axes[2].plot(t_span, x, 'r-', linewidth=2.5)
    axes[2].set_xlabel('Время, с', fontsize=11)
    axes[2].set_ylabel('Положение x, м', fontsize=11)
    axes[2].set_title('Горизонтальное положение', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Сохранение графика
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/report/images"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "free_motion.png"), dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_dir}/free_motion.png")
    plt.close()
    
    # Фазовый портрет (theta vs theta_dot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Используем цветовую кодировку по времени для лучшей визуализации
    # и избегаем соединения начала и конца
    theta_deg = np.rad2deg(theta)
    theta_dot_deg = np.rad2deg(theta_dot)
    
    # Рисуем траекторию с цветовой кодировкой по времени
    scatter = ax.scatter(theta_deg, theta_dot_deg, c=t_span, cmap='viridis', 
                        s=2, alpha=0.6, zorder=1)
    
    # Рисуем основную линию траектории
    ax.plot(theta_deg, theta_dot_deg, 'b-', linewidth=1.0, alpha=0.4, zorder=2)
    
    # Добавляем стрелки направления в нескольких точках
    arrow_indices = np.linspace(0, len(theta)-1, 20, dtype=int)
    for i in arrow_indices[1:-1]:  # Пропускаем первый и последний
        dx = theta_deg[i+1] - theta_deg[i]
        dy = theta_dot_deg[i+1] - theta_dot_deg[i]
        if abs(dx) > 0.1 or abs(dy) > 0.1:  # Рисуем стрелку только если есть движение
            ax.arrow(theta_deg[i], theta_dot_deg[i], dx*0.3, dy*0.3,
                    head_width=0.5, head_length=0.3, fc='red', ec='red', 
                    alpha=0.6, zorder=3, length_includes_head=True)
    
    # Начальная точка
    ax.scatter(theta_deg[0], theta_dot_deg[0], color='green', s=150, 
               marker='o', label='Начало', zorder=5, edgecolors='black', linewidths=2)
    
    # Конечная точка
    ax.scatter(theta_deg[-1], theta_dot_deg[-1], color='red', s=150, 
               marker='x', label='Конец', zorder=5, linewidths=3)
    
    # Добавляем цветовую шкалу времени
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Время, с', rotation=270, labelpad=15)
    
    ax.set_xlabel('Угол наклона θ, град')
    ax.set_ylabel('Угловая скорость θ̇, град/с')
    ax.set_title('Фазовый портрет: угол наклона платформы (свободное движение)\n' +
                 'Цвет показывает время, стрелки - направление движения')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.legend(loc='best')
    
    plt.savefig(os.path.join(output_dir, "free_motion_phase.png"), dpi=300, bbox_inches='tight')
    print(f"Фазовый портрет сохранен: {output_dir}/free_motion_phase.png")
    plt.close()
    
    # Дополнительная диагностика
    print(f"\n=== Диагностика фазового портрета ===")
    print(f"Начальная точка: θ={theta_deg[0]:.2f}°, θ̇={theta_dot_deg[0]:.2f}°/с")
    print(f"Конечная точка: θ={theta_deg[-1]:.2f}°, θ̇={theta_dot_deg[-1]:.2f}°/с")
    print(f"Расстояние от начала координат в начале: {np.sqrt(theta_deg[0]**2 + theta_dot_deg[0]**2):.2f}")
    print(f"Расстояние от начала координат в конце: {np.sqrt(theta_deg[-1]**2 + theta_dot_deg[-1]**2):.2f}")
    print(f"Максимальное расстояние от начала координат: {np.max(np.sqrt(theta_deg**2 + theta_dot_deg**2)):.2f}")
    
    # Анализ результатов
    print("\n=== Анализ свободного движения ===")
    print(f"Начальное отклонение: {np.rad2deg(theta0):.2f} град")
    print(f"Максимальное отклонение от вертикали: {np.max(np.abs(np.rad2deg(theta))):.2f} град")
    print(f"Рост отклонения: {np.max(np.abs(np.rad2deg(theta)))/np.abs(np.rad2deg(theta0)):.2f}x")
    print(f"Время до падения (|theta| > 45°): ", end="")
    fall_idx = np.where(np.abs(theta) > np.deg2rad(45))[0]
    if len(fall_idx) > 0:
        print(f"{t_span[fall_idx[0]]:.2f} с")
        print(f"✓ Система упала за время симуляции")
    else:
        print("не достигнуто за время симуляции")
        print(f"  (максимальное отклонение: {np.max(np.abs(np.rad2deg(theta))):.2f}°)")
    print(f"Максимальная горизонтальная скорость: {np.max(np.abs(x_dot)):.3f} м/с")
    print(f"Максимальное смещение по горизонтали: {np.max(np.abs(x)):.3f} м")

if __name__ == "__main__":
    main()

