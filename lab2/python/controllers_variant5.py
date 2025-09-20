import numpy as np
import matplotlib.pyplot as plt
import os

def simple_static_controller_variant5(state, ref_state, ref_vel, kp=1.0, kd=0.5):
    """
    Простой статический контроллер для варианта 5
    """
    x, y, theta, vx, vy, omega = state
    x_ref, y_ref, theta_ref = ref_state
    vx_ref, vy_ref, omega_ref = ref_vel
    
    # Ошибки
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Простое управление
    ux = vx_ref + kp * ex
    uy = vy_ref + kp * ey
    utheta = omega_ref + kp * etheta
    
    return np.array([ux, uy, utheta])

def simple_dynamic_controller_variant5(state, ref_state, ref_vel, ref_acc, kp=1.0, kd=0.5, ki=0.1, integral_errors=None):
    """
    Простой динамический контроллер с интегральным управлением для варианта 5
    """
    if integral_errors is None:
        integral_errors = np.zeros(3)
    
    x, y, theta, vx, vy, omega = state
    x_ref, y_ref, theta_ref = ref_state
    vx_ref, vy_ref, omega_ref = ref_vel
    ax_ref, ay_ref, alpha_ref = ref_acc
    
    # Ошибки
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Обновление интегральных ошибок
    integral_errors[0] += ex
    integral_errors[1] += ey
    integral_errors[2] += etheta
    
    # Динамическое управление
    ux = ax_ref + kp * ex + kd * (vx_ref - vx) + ki * integral_errors[0]
    uy = ay_ref + kp * ey + kd * (vy_ref - vy) + ki * integral_errors[1]
    utheta = alpha_ref + kp * etheta + kd * (omega_ref - omega) + ki * integral_errors[2]
    
    return np.array([ux, uy, utheta]), integral_errors

def simulate_robot_tracking_variant5(x0, t, ref_trajectory, controller_type='static', kp=1.0, kd=0.5, ki=0.1):
    """
    Симуляция слежения за траекторией для варианта 5
    """
    dt = t[1] - t[0]
    n = len(t)
    
    # Инициализация
    state = x0.copy()
    states = np.zeros((n, 6))
    states[0] = state
    
    integral_errors = np.zeros(3)
    
    for i in range(1, n):
        # Получение эталонной траектории
        ref_state, ref_vel, ref_acc = ref_trajectory(t[i])
        
        # Вычисление управления
        if controller_type == 'static':
            u = simple_static_controller_variant5(state, ref_state, ref_vel, kp, kd)
        else:
            u, integral_errors = simple_dynamic_controller_variant5(state, ref_state, ref_vel, ref_acc, 
                                                                 kp, kd, ki, integral_errors)
        
        # Простая интеграция (метод Эйлера)
        # Упрощенная динамика: x' = ux, y' = uy, theta' = utheta
        state[0] += u[0] * dt  # x
        state[1] += u[1] * dt  # y
        state[2] += u[2] * dt  # theta
        state[3] = u[0]  # vx
        state[4] = u[1]  # vy
        state[5] = u[2]  # omega
        
        states[i] = state
    
    return states

def create_reference_trajectory_variant5(x_traj, y_traj, theta_traj, t):
    """
    Создание функции эталонной траектории для варианта 5
    """
    from scipy.interpolate import interp1d
    
    # Интерполяция позиций
    x_ref_func = interp1d(t, x_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_ref_func = interp1d(t, y_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    theta_ref_func = interp1d(t, theta_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Вычисление скоростей
    vx_ref = np.gradient(x_traj, t)
    vy_ref = np.gradient(y_traj, t)
    omega_ref = np.gradient(theta_traj, t)
    
    vx_ref_func = interp1d(t, vx_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    vy_ref_func = interp1d(t, vy_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    omega_ref_func = interp1d(t, omega_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Вычисление ускорений
    ax_ref = np.gradient(vx_ref, t)
    ay_ref = np.gradient(vy_ref, t)
    alpha_ref = np.gradient(omega_ref, t)
    
    ax_ref_func = interp1d(t, ax_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    ay_ref_func = interp1d(t, ay_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    alpha_ref_func = interp1d(t, alpha_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    def ref_trajectory(t):
        ref_state = np.array([x_ref_func(t), y_ref_func(t), theta_ref_func(t)])
        ref_vel = np.array([vx_ref_func(t), vy_ref_func(t), omega_ref_func(t)])
        ref_acc = np.array([ax_ref_func(t), ay_ref_func(t), alpha_ref_func(t)])
        return ref_state, ref_vel, ref_acc
    
    return ref_trajectory

def plot_comparison_variant5(t, x_ref, y_ref, theta_ref, 
                          x_static, y_static, theta_static,
                          x_dynamic, y_dynamic, theta_dynamic,
                          title="Сравнение методов линеаризации (вариант 5)"):
    """Построение графиков сравнения результатов для варианта 5"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Траектория в плоскости
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, label='Эталонная траектория')
    ax1.plot(x_static, y_static, 'b-', linewidth=2, label='Статическая линеаризация')
    ax1.plot(x_dynamic, y_dynamic, 'r-', linewidth=2, label='Динамическая линеаризация')
    ax1.plot(x_ref[0], y_ref[0], 'go', markersize=8, label='Начало')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектория в плоскости (вариант 5)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Ошибки позиции
    ex_static = x_ref - x_static
    ey_static = y_ref - y_static
    ex_dynamic = x_ref - x_dynamic
    ey_dynamic = y_ref - y_dynamic
    
    ax2.plot(t, ex_static, 'b-', linewidth=2, label='Ошибка X (статическая)')
    ax2.plot(t, ey_static, 'b--', linewidth=2, label='Ошибка Y (статическая)')
    ax2.plot(t, ex_dynamic, 'r-', linewidth=2, label='Ошибка X (динамическая)')
    ax2.plot(t, ey_dynamic, 'r--', linewidth=2, label='Ошибка Y (динамическая)')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Ошибка позиции (м)')
    ax2.set_title('Ошибки позиции (вариант 5)')
    ax2.legend()
    ax2.grid(True)
    
    # Угол поворота
    ax3.plot(t, theta_ref, 'k--', linewidth=2, label='Эталонный угол')
    ax3.plot(t, theta_static, 'b-', linewidth=2, label='Статическая линеаризация')
    ax3.plot(t, theta_dynamic, 'r-', linewidth=2, label='Динамическая линеаризация')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Угол поворота (рад)')
    ax3.set_title('Ориентация робота (вариант 5)')
    ax3.legend()
    ax3.grid(True)
    
    # Ошибка угла
    etheta_static = theta_ref - theta_static
    etheta_dynamic = theta_ref - theta_dynamic
    ax4.plot(t, etheta_static, 'b-', linewidth=2, label='Ошибка угла (статическая)')
    ax4.plot(t, etheta_dynamic, 'r-', linewidth=2, label='Ошибка угла (динамическая)')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Ошибка угла (рад)')
    ax4.set_title('Ошибка ориентации (вариант 5)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    from robot_model_variant5 import generate_trajectory_variant5
    
    # Временной интервал
    t = np.linspace(0, 30, 1500)
    
    # Генерация эталонной траектории для варианта 5
    x_ref, y_ref, theta_ref = generate_trajectory_variant5(t, R1=7.0, R2=12.0, 
                                                         alpha=np.pi/6, delta=2*np.pi, 
                                                         t_straight=6.0)
    
    # Создание функции эталонной траектории
    ref_trajectory = create_reference_trajectory_variant5(x_ref, y_ref, theta_ref, t)
    
    # Начальное состояние для варианта 5: [0, 3, 2π/3]
    x0 = np.array([0.1, 3.1, 2*np.pi/3 + 0.05, 0.0, 0.0, 0.0])
    
    # Симуляция статической линеаризации
    states_static = simulate_robot_tracking_variant5(x0, t, ref_trajectory, 'static', kp=1.0, kd=0.5)
    
    # Симуляция динамической линеаризации
    states_dynamic = simulate_robot_tracking_variant5(x0, t, ref_trajectory, 'dynamic', kp=1.0, kd=0.5, ki=0.1)
    
    # Извлечение результатов
    x_static = states_static[:, 0]
    y_static = states_static[:, 1]
    theta_static = states_static[:, 2]
    
    x_dynamic = states_dynamic[:, 0]
    y_dynamic = states_dynamic[:, 1]
    theta_dynamic = states_dynamic[:, 2]
    
    # Построение графиков
    fig = plot_comparison_variant5(t, x_ref, y_ref, theta_ref,
                                x_static, y_static, theta_static,
                                x_dynamic, y_dynamic, theta_dynamic)
    
    # Сохранение результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab2/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, "controllers_comparison_variant5.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение данных
    np.savez(os.path.join(output_dir, "controllers_variant5_results.npz"),
             t=t, x_ref=x_ref, y_ref=y_ref, theta_ref=theta_ref,
             x_static=x_static, y_static=y_static, theta_static=theta_static,
             x_dynamic=x_dynamic, y_dynamic=y_dynamic, theta_dynamic=theta_dynamic)
    
    # Вычисление ошибок
    max_error_x_static = np.max(np.abs(x_ref - x_static))
    max_error_y_static = np.max(np.abs(y_ref - y_static))
    max_error_theta_static = np.max(np.abs(theta_ref - theta_static))
    
    max_error_x_dynamic = np.max(np.abs(x_ref - x_dynamic))
    max_error_y_dynamic = np.max(np.abs(y_ref - y_dynamic))
    max_error_theta_dynamic = np.max(np.abs(theta_ref - theta_dynamic))
    
    print(f"Результаты сравнения контроллеров для варианта 5 сохранены в {output_dir}")
    print("\nСтатическая линеаризация:")
    print(f"  Максимальная ошибка по X: {max_error_x_static:.4f} м")
    print(f"  Максимальная ошибка по Y: {max_error_y_static:.4f} м")
    print(f"  Максимальная ошибка по углу: {max_error_theta_static:.4f} рад")
    
    print("\nДинамическая линеаризация:")
    print(f"  Максимальная ошибка по X: {max_error_x_dynamic:.4f} м")
    print(f"  Максимальная ошибка по Y: {max_error_y_dynamic:.4f} м")
    print(f"  Максимальная ошибка по углу: {max_error_theta_dynamic:.4f} рад")

if __name__ == "__main__":
    main()
