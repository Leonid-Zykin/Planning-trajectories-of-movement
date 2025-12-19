"""
Симуляция управляемого движения Segway с контроллером по выходу
Метод управления: стабилизация по выходу (output feedback)
Выходные переменные: y = [theta, theta_dot] (угол наклона и его производная)
"""

import numpy as np
import matplotlib.pyplot as plt
from segway_model import SegwayModel
import os

class OutputFeedbackController:
    """
    Контроллер по выходу для стабилизации Segway.
    
    Использует только измеряемые выходные переменные:
    y = [theta, theta_dot] - угол наклона и его производная
    
    Закон управления: u = -K_p * theta - K_d * theta_dot
    где K_p, K_d > 0 - коэффициенты обратной связи
    """
    
    def __init__(self, K_p=50.0, K_d=10.0):
        """
        Args:
            K_p: пропорциональный коэффициент (усиление по углу)
            K_d: дифференциальный коэффициент (усиление по угловой скорости)
        """
        self.K_p = K_p
        self.K_d = K_d
        
    def control_law(self, state, t):
        """
        Вычисление управляющего воздействия по выходу.
        
        Args:
            state: [theta, theta_dot, phi, phi_dot, x, x_dot]
            t: время
            
        Returns:
            u: управляющий момент
        """
        theta = state[0]
        theta_dot = state[1]
        
        # Контроллер по выходу: используем только theta и theta_dot
        u = -self.K_p * theta - self.K_d * theta_dot
        
        # Ограничение управляющего воздействия
        u_max = 100.0  # Максимальный момент (Н·м)
        u = np.clip(u, -u_max, u_max)
        
        return u

def main():
    # Создание модели
    segway = SegwayModel(M=10.0, m=2.0, L=0.5, R=0.1, I_p=1.0, I_w=0.1, g=9.81)
    
    # Создание контроллера по выходу
    # Подобраны коэффициенты для стабилизации
    controller = OutputFeedbackController(K_p=50.0, K_d=10.0)
    
    # Начальные условия: отклонение от вертикали
    theta0 = np.deg2rad(10.0)  # 10 градусов от вертикали
    theta_dot0 = 0.0
    phi0 = 0.0
    phi_dot0 = 0.0
    x0 = 0.0
    x_dot0 = 0.0
    
    initial_state = [theta0, theta_dot0, phi0, phi_dot0, x0, x_dot0]
    
    # Временной интервал
    t_span = np.linspace(0, 10.0, 10000)
    
    # Симуляция управляемого движения
    print("Симуляция управляемого движения Segway с контроллером по выходу...")
    sol = segway.simulate_controlled(initial_state, t_span, controller.control_law)
    
    theta = sol[:, 0]
    theta_dot = sol[:, 1]
    phi = sol[:, 2]
    phi_dot = sol[:, 3]
    x = sol[:, 4]
    x_dot = sol[:, 5]
    
    # Вычисление управляющих воздействий
    u_history = []
    for i, t in enumerate(t_span):
        u = controller.control_law(sol[i], t)
        u_history.append(u)
    u_history = np.array(u_history)
    
    # Построение графиков (расширенная картинка для отчёта)
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Угол наклона платформы
    axes[0, 0].plot(t_span, np.rad2deg(theta), 'b-', linewidth=2, label='Управляемое движение')
    axes[0, 0].set_xlabel('Время, с')
    axes[0, 0].set_ylabel('Угол наклона θ, град')
    axes[0, 0].set_title('Угол наклона платформы (управляемое движение)')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Целевое положение')
    axes[0, 0].legend()
    
    # Угловая скорость наклона
    axes[0, 1].plot(t_span, np.rad2deg(theta_dot), 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Время, с')
    axes[0, 1].set_ylabel('Угловая скорость θ̇, град/с')
    axes[0, 1].set_title('Угловая скорость наклона платформы')
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Управляющее воздействие
    axes[1, 0].plot(t_span, u_history, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Время, с')
    axes[1, 0].set_ylabel('Управляющий момент u, Н·м')
    axes[1, 0].set_title('Управляющее воздействие')
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Горизонтальное положение
    axes[1, 1].plot(t_span, x, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Время, с')
    axes[1, 1].set_ylabel('Положение x, м')
    axes[1, 1].set_title('Горизонтальное положение')
    axes[1, 1].grid(True)
    
    # Фазовый портрет (theta vs theta_dot)
    axes[2, 0].plot(np.rad2deg(theta), np.rad2deg(theta_dot), 'b-', linewidth=1.5, alpha=0.7)
    axes[2, 0].scatter(np.rad2deg(theta[0]), np.rad2deg(theta_dot[0]), color='green', s=100, 
                       marker='o', label='Начало', zorder=5)
    axes[2, 0].scatter(0, 0, color='red', s=100, marker='x', label='Цель', zorder=5)
    axes[2, 0].set_xlabel('Угол наклона θ, град')
    axes[2, 0].set_ylabel('Угловая скорость θ̇, град/с')
    axes[2, 0].set_title('Фазовый портрет: угол наклона платформы')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()
    
    # Ошибка стабилизации (используется только в расширенной картинке для отчёта)
    error = np.abs(theta)
    error_deg = np.rad2deg(error)
    error_log = np.maximum(error_deg, 1e-8)
    axes[2, 1].semilogy(t_span, error_log, 'm-', linewidth=2, label='Ошибка |θ|')
    axes[2, 1].set_xlabel('Время, с')
    axes[2, 1].set_ylabel('Ошибка |θ|, град (логарифмическая шкала)')
    axes[2, 1].set_title('Ошибка стабилизации (логарифмическая шкала)')
    axes[2, 1].grid(True, which='both', alpha=0.3)
    axes[2, 1].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Сохранение расширенного графика (для отчёта)
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/report/images"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "controlled_motion.png"), dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_dir}/controlled_motion.png")
    plt.close()

    # Упрощённый график для презентации (без фазового портрета и ошибки стабилизации)
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))

    # Угол наклона
    axes2[0, 0].plot(t_span, np.rad2deg(theta), 'b-', linewidth=2)
    axes2[0, 0].set_xlabel('Время, с')
    axes2[0, 0].set_ylabel('Угол θ, град')
    axes2[0, 0].set_title('Угол наклона платформы')
    axes2[0, 0].grid(True)

    # Угловая скорость
    axes2[0, 1].plot(t_span, np.rad2deg(theta_dot), 'b-', linewidth=2)
    axes2[0, 1].set_xlabel('Время, с')
    axes2[0, 1].set_ylabel('Угловая скорость θ̇, град/с')
    axes2[0, 1].set_title('Угловая скорость наклона')
    axes2[0, 1].grid(True)

    # Управляющий момент
    axes2[1, 0].plot(t_span, u_history, 'g-', linewidth=2)
    axes2[1, 0].set_xlabel('Время, с')
    axes2[1, 0].set_ylabel('Момент u, Н·м')
    axes2[1, 0].set_title('Управляющее воздействие')
    axes2[1, 0].grid(True)

    # Горизонтальное положение
    axes2[1, 1].plot(t_span, x, 'r-', linewidth=2)
    axes2[1, 1].set_xlabel('Время, с')
    axes2[1, 1].set_ylabel('Положение x, м')
    axes2[1, 1].set_title('Горизонтальное положение')
    axes2[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "controlled_motion_presentation.png"), dpi=300, bbox_inches='tight')
    print(f"График для презентации сохранен: {output_dir}/controlled_motion_presentation.png")
    plt.close()
    
    # Сравнительный график: свободное vs управляемое движение
    # Симуляция свободного движения для сравнения
    sol_free = segway.simulate_free(initial_state, t_span)
    theta_free = sol_free[:, 0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_span, np.rad2deg(theta), 'b-', linewidth=2, label='Управляемое движение')
    ax.plot(t_span, np.rad2deg(theta_free), 'r--', linewidth=2, label='Свободное движение')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Угол наклона θ, град')
    ax.set_title('Сравнение свободного и управляемого движения')
    ax.grid(True)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend()
    plt.savefig(os.path.join(output_dir, "comparison_free_vs_controlled.png"), dpi=300, bbox_inches='tight')
    print(f"Сравнительный график сохранен: {output_dir}/comparison_free_vs_controlled.png")
    plt.close()
    
    # Анализ результатов
    print("\n=== Анализ управляемого движения ===")
    print(f"Параметры контроллера: K_p = {controller.K_p}, K_d = {controller.K_d}")
    print(f"Максимальное отклонение: {np.max(np.abs(np.rad2deg(theta))):.4f} град")
    print(f"Установившаяся ошибка (последние 2 секунды): {np.mean(np.abs(np.rad2deg(theta[-2000:]))):.6f} град")
    print(f"Время установления (до 1°): ", end="")
    settle_idx = np.where(np.abs(theta) < np.deg2rad(1.0))[0]
    if len(settle_idx) > 0:
        print(f"{t_span[settle_idx[0]]:.3f} с")
    else:
        print("не достигнуто")
    print(f"Максимальное управляющее воздействие: {np.max(np.abs(u_history)):.2f} Н·м")
    print(f"Средняя мощность управления: {np.mean(np.abs(u_history * phi_dot)):.2f} Вт")

if __name__ == "__main__":
    main()

