import numpy as np
import matplotlib.pyplot as plt
import os

def simple_static_controller(state, ref_state, ref_vel, kp=0.05):
    """
    Простой статический контроллер для четырехколесного робота с дифференциальным приводом
    """
    x, y, theta = state
    x_ref, y_ref, theta_ref = ref_state
    v_ref, omega_ref = ref_vel
    
    # Ошибки
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Простое управление с малыми коэффициентами
    v = v_ref + kp * ex
    omega = omega_ref + kp * etheta
    
    return np.array([v, omega])

def simple_dynamic_controller(state, ref_state, ref_vel, kp=0.05, kd=0.01, ki=0.001, integral_errors=None):
    """
    Простой динамический контроллер для четырехколесного робота
    """
    if integral_errors is None:
        integral_errors = np.zeros(2)
    
    x, y, theta = state
    x_ref, y_ref, theta_ref = ref_state
    v_ref, omega_ref = ref_vel
    
    # Ошибки
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Простые производные ошибок
    dex_dt = v_ref * np.cos(theta_ref) - v_ref * np.cos(theta)
    detheta_dt = omega_ref - omega_ref
    
    # Интегральные ошибки
    integral_errors[0] += ex
    integral_errors[1] += etheta
    
    # ПИД управление
    v = v_ref + kp * ex + kd * dex_dt + ki * integral_errors[0]
    omega = omega_ref + kp * etheta + kd * detheta_dt + ki * integral_errors[1]
    
    return np.array([v, omega]), integral_errors

def simulate_simple_controllers(trajectory_data, dt=0.01):
    """
    Простая симуляция контроллеров
    """
    x_traj = trajectory_data['x']
    y_traj = trajectory_data['y']
    theta_traj = trajectory_data['theta']
    
    n_points = len(x_traj)
    time_points = np.arange(0, n_points * dt, dt)
    
    # Начальное состояние
    state_static = np.array([x_traj[0], y_traj[0], theta_traj[0]])
    state_dynamic = np.array([x_traj[0], y_traj[0], theta_traj[0]])
    
    # Массивы для результатов
    states_static = [state_static.copy()]
    states_dynamic = [state_dynamic.copy()]
    errors_static = []
    errors_dynamic = []
    
    integral_errors = np.zeros(2)
    
    for i in range(1, min(n_points, 1000)):  # Ограничиваем количество точек
        # Эталонные значения
        ref_state = np.array([x_traj[i], y_traj[i], theta_traj[i]])
        
        # Простые эталонные скорости
        if i > 0:
            v_ref = 2.0  # Постоянная скорость
            omega_ref = 0.1  # Небольшая угловая скорость
        else:
            v_ref = 0
            omega_ref = 0
        
        ref_vel = np.array([v_ref, omega_ref])
        
        # Статический контроллер
        control_static = simple_static_controller(state_static, ref_state, ref_vel)
        
        # Динамический контроллер
        control_dynamic, integral_errors = simple_dynamic_controller(
            state_dynamic, ref_state, ref_vel, integral_errors=integral_errors
        )
        
        # Обновление состояний с ограничениями
        v_static = np.clip(control_static[0], -3, 3)
        omega_static = np.clip(control_static[1], -1, 1)
        
        state_static += np.array([
            v_static * np.cos(state_static[2]) * dt,
            v_static * np.sin(state_static[2]) * dt,
            omega_static * dt
        ])
        
        v_dynamic = np.clip(control_dynamic[0], -3, 3)
        omega_dynamic = np.clip(control_dynamic[1], -1, 1)
        
        state_dynamic += np.array([
            v_dynamic * np.cos(state_dynamic[2]) * dt,
            v_dynamic * np.sin(state_dynamic[2]) * dt,
            omega_dynamic * dt
        ])
        
        states_static.append(state_static.copy())
        states_dynamic.append(state_dynamic.copy())
        
        # Ошибки
        error_static = np.linalg.norm(state_static[:2] - ref_state[:2])
        error_dynamic = np.linalg.norm(state_dynamic[:2] - ref_state[:2])
        
        errors_static.append(error_static)
        errors_dynamic.append(error_dynamic)
    
    return {
        'time': time_points[:len(states_static)],
        'states_static': np.array(states_static),
        'states_dynamic': np.array(states_dynamic),
        'errors_static': np.array(errors_static),
        'errors_dynamic': np.array(errors_dynamic),
        'reference': np.column_stack([x_traj[:len(states_static)], 
                                     y_traj[:len(states_static)], 
                                     theta_traj[:len(states_static)]])
    }

def plot_simple_comparison(simulation_results, save_path=None):
    """
    Простое сравнение контроллеров
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    time = simulation_results['time']
    states_static = simulation_results['states_static']
    states_dynamic = simulation_results['states_dynamic']
    reference = simulation_results['reference']
    errors_static = simulation_results['errors_static']
    errors_dynamic = simulation_results['errors_dynamic']
    
    # Траектории
    ax1 = axes[0, 0]
    ax1.plot(reference[:, 0], reference[:, 1], 'k--', linewidth=2, label='Эталонная траектория')
    ax1.plot(states_static[:, 0], states_static[:, 1], 'b-', linewidth=1, label='Статическая линеаризация')
    ax1.plot(states_dynamic[:, 0], states_dynamic[:, 1], 'r-', linewidth=1, label='Динамическая линеаризация')
    ax1.set_xlabel('X, м')
    ax1.set_ylabel('Y, м')
    ax1.set_title('Траектории движения четырехколесного робота с дифференциальным приводом')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Ошибки
    ax2 = axes[0, 1]
    ax2.plot(time[:-1], errors_static, 'b-', linewidth=2, label='Статическая линеаризация')
    ax2.plot(time[:-1], errors_dynamic, 'r-', linewidth=2, label='Динамическая линеаризация')
    ax2.set_xlabel('Время, с')
    ax2.set_ylabel('Ошибка позиции, м')
    ax2.set_title('Ошибки слежения')
    ax2.legend()
    ax2.grid(True)
    
    # Ошибки по X
    ax3 = axes[1, 0]
    error_x_static = reference[:-1, 0] - states_static[:-1, 0]
    error_x_dynamic = reference[:-1, 0] - states_dynamic[:-1, 0]
    ax3.plot(time[:-1], error_x_static, 'b-', linewidth=2, label='Статическая линеаризация')
    ax3.plot(time[:-1], error_x_dynamic, 'r-', linewidth=2, label='Динамическая линеаризация')
    ax3.set_xlabel('Время, с')
    ax3.set_ylabel('Ошибка по X, м')
    ax3.set_title('Ошибки по координате X')
    ax3.legend()
    ax3.grid(True)
    
    # Ошибки по Y
    ax4 = axes[1, 1]
    error_y_static = reference[:-1, 1] - states_static[:-1, 1]
    error_y_dynamic = reference[:-1, 1] - states_dynamic[:-1, 1]
    ax4.plot(time[:-1], error_y_static, 'b-', linewidth=2, label='Статическая линеаризация')
    ax4.plot(time[:-1], error_y_dynamic, 'r-', linewidth=2, label='Динамическая линеаризация')
    ax4.set_xlabel('Время, с')
    ax4.set_ylabel('Ошибка по Y, м')
    ax4.set_title('Ошибки по координате Y')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_simple_error_analysis(simulation_results, save_path=None):
    """
    Простой анализ ошибок
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    time = simulation_results['time']
    states_static = simulation_results['states_static']
    states_dynamic = simulation_results['states_dynamic']
    reference = simulation_results['reference']
    
    # Ошибки по X
    error_x_static = reference[:-1, 0] - states_static[:-1, 0]
    error_x_dynamic = reference[:-1, 0] - states_dynamic[:-1, 0]
    
    axes[0].plot(time[:-1], error_x_static, 'b-', linewidth=2, label='Статическая линеаризация')
    axes[0].plot(time[:-1], error_x_dynamic, 'r--', linewidth=2, label='Динамическая линеаризация')
    axes[0].set_ylabel('Ошибка по X, м')
    axes[0].set_title('Анализ ошибок слежения за траекторией (четырехколесный робот с дифференциальным приводом)')
    axes[0].grid(True)
    axes[0].legend()
    
    # Ошибки по Y
    error_y_static = reference[:-1, 1] - states_static[:-1, 1]
    error_y_dynamic = reference[:-1, 1] - states_dynamic[:-1, 1]
    
    axes[1].plot(time[:-1], error_y_static, 'b-', linewidth=2, label='Статическая линеаризация')
    axes[1].plot(time[:-1], error_y_dynamic, 'r--', linewidth=2, label='Динамическая линеаризация')
    axes[1].set_ylabel('Ошибка по Y, м')
    axes[1].grid(True)
    axes[1].legend()
    
    # Ошибки по углу
    error_theta_static = reference[:-1, 2] - states_static[:-1, 2]
    error_theta_dynamic = reference[:-1, 2] - states_dynamic[:-1, 2]
    
    axes[2].plot(time[:-1], error_theta_static, 'b-', linewidth=2, label='Статическая линеаризация')
    axes[2].plot(time[:-1], error_theta_dynamic, 'r--', linewidth=2, label='Динамическая линеаризация')
    axes[2].set_xlabel('Время, с')
    axes[2].set_ylabel('Ошибка по углу, рад')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def main():
    """
    Основная функция
    """
    # Загрузка данных траектории
    trajectory_file = "images/task1/trajectory_variant5_data.npz"
    if os.path.exists(trajectory_file):
        trajectory_data = np.load(trajectory_file)
        print(f"Загружены данные траектории: {trajectory_file}")
    else:
        print("Данные траектории не найдены. Сначала запустите robot_model_variant5.py")
        return
    
    # Симуляция
    print("Запуск простой симуляции контроллеров...")
    simulation_results = simulate_simple_controllers(trajectory_data)
    
    # Построение графиков
    output_dir = "images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    # График сравнения
    controllers_file = os.path.join(output_dir, "controllers_comparison_variant5.png")
    plot_simple_comparison(simulation_results, controllers_file)
    print(f"График сравнения контроллеров сохранен: {controllers_file}")
    
    # График анализа ошибок
    error_file = os.path.join(output_dir, "error_analysis.png")
    plot_simple_error_analysis(simulation_results, error_file)
    print(f"График анализа ошибок сохранен: {error_file}")
    
    # Статистика
    print("\n=== СТАТИСТИКА ОШИБОК ===")
    print(f"Статическая линеаризация:")
    print(f"  Максимальная ошибка: {np.max(simulation_results['errors_static']):.4f} м")
    print(f"  Средняя ошибка: {np.mean(simulation_results['errors_static']):.4f} м")
    print(f"  СКО ошибки: {np.std(simulation_results['errors_static']):.4f} м")
    
    print(f"Динамическая линеаризация:")
    print(f"  Максимальная ошибка: {np.max(simulation_results['errors_dynamic']):.4f} м")
    print(f"  Средняя ошибка: {np.mean(simulation_results['errors_dynamic']):.4f} м")
    print(f"  СКО ошибки: {np.std(simulation_results['errors_dynamic']):.4f} м")

if __name__ == "__main__":
    main()