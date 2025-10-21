import numpy as np
import matplotlib.pyplot as plt
import os

def static_linearization_controller_type12(state, ref_state, ref_eta, kp1=1.0, kp2=1.0, kp_beta=0.5):
    """
    Статическая линеаризация для робота типа (1,2)
    state = [x, y, theta, beta_s1, beta_s2, beta_c3]
    ref_state = [x_ref, y_ref, theta_ref]
    ref_eta = [eta1_ref, eta2_ref]
    """
    x, y, theta, beta_s1, beta_s2, beta_c3 = state
    x_ref, y_ref, theta_ref = ref_state
    eta1_ref, eta2_ref = ref_eta
    
    # Ошибки позиции и ориентации
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Ошибки углов поворота колес
    e_beta_s1 = 0 - beta_s1  # Эталонный угол для прямолинейного движения
    e_beta_s2 = 0 - beta_s2
    
    # Законы управления для обобщенных скоростей
    eta1 = eta1_ref + kp1 * ex
    eta2 = eta2_ref + kp2 * ey
    
    # Законы управления для углов поворота колес
    zeta1 = kp_beta * e_beta_s1
    zeta2 = kp_beta * e_beta_s2
    
    return np.array([eta1, eta2, zeta1, zeta2])

def dynamic_linearization_controller_type12(state, ref_state, ref_eta, ref_zeta, 
                                          kp1=1.0, kp2=1.0, kp_beta=0.5,
                                          kd1=0.5, kd2=0.5, kd_beta=0.2,
                                          ki1=0.1, ki2=0.1, ki_beta=0.05, 
                                          integral_errors=None):
    """
    Динамическая линеаризация с ПИД-управлением для робота типа (1,2)
    """
    if integral_errors is None:
        integral_errors = np.zeros(5)  # Для eta1, eta2, beta_s1, beta_s2, theta
    
    x, y, theta, beta_s1, beta_s2, beta_c3 = state
    x_ref, y_ref, theta_ref = ref_state
    eta1_ref, eta2_ref = ref_eta
    zeta1_ref, zeta2_ref = ref_zeta
    
    # Ошибки позиции и ориентации
    ex = x_ref - x
    ey = y_ref - y
    etheta = theta_ref - theta
    
    # Ошибки углов поворота колес
    e_beta_s1 = 0 - beta_s1
    e_beta_s2 = 0 - beta_s2
    
    # Простые производные ошибок (упрощенно)
    dex_dt = eta1_ref - eta1_ref  # Упрощенно
    dey_dt = eta2_ref - eta2_ref
    detheta_dt = 0  # Упрощенно
    
    # Обновление интегральных ошибок
    integral_errors[0] += ex  # Интеграл ошибки по x
    integral_errors[1] += ey  # Интеграл ошибки по y
    integral_errors[2] += etheta  # Интеграл ошибки по theta
    integral_errors[3] += e_beta_s1  # Интеграл ошибки по beta_s1
    integral_errors[4] += e_beta_s2  # Интеграл ошибки по beta_s2
    
    # ПИД управление для обобщенных скоростей
    eta1 = eta1_ref + kp1 * ex + kd1 * dex_dt + ki1 * integral_errors[0]
    eta2 = eta2_ref + kp2 * ey + kd2 * dey_dt + ki2 * integral_errors[1]
    
    # ПИД управление для углов поворота колес
    zeta1 = zeta1_ref + kp_beta * e_beta_s1 + kd_beta * (0 - 0) + ki_beta * integral_errors[3]
    zeta2 = zeta2_ref + kp_beta * e_beta_s2 + kd_beta * (0 - 0) + ki_beta * integral_errors[4]
    
    return np.array([eta1, eta2, zeta1, zeta2]), integral_errors

def simulate_robot_tracking_type12(x0, t, ref_trajectory, controller_type='static', 
                                  kp1=1.0, kp2=1.0, kp_beta=0.5, kd1=0.5, kd2=0.5, kd_beta=0.2,
                                  ki1=0.1, ki2=0.1, ki_beta=0.05):
    """
    Симуляция слежения за траекторией для робота типа (1,2)
    """
    dt = t[1] - t[0]
    n = len(t)
    
    # Инициализация
    state = x0.copy()
    states = np.zeros((n, 6))
    states[0] = state
    
    integral_errors = np.zeros(5)
    
    for i in range(1, n):
        # Получение эталонной траектории
        ref_state, ref_eta, ref_zeta = ref_trajectory(t[i])
        
        # Вычисление управления
        if controller_type == 'static':
            u = static_linearization_controller_type12(state, ref_state, ref_eta, kp1, kp2, kp_beta)
        else:
            u, integral_errors = dynamic_linearization_controller_type12(
                state, ref_state, ref_eta, ref_zeta, kp1, kp2, kp_beta, 
                kd1, kd2, kd_beta, ki1, ki2, ki_beta, integral_errors
            )
        
        # Простая интеграция (метод Эйлера)
        # Упрощенная кинематическая модель
        eta1, eta2, zeta1, zeta2 = u
        
        # Обновление состояния
        state[0] += eta1 * np.cos(state[2]) * dt  # x
        state[1] += eta1 * np.sin(state[2]) * dt  # y
        state[2] += eta2 * dt  # theta
        state[3] += zeta1 * dt  # beta_s1
        state[4] += zeta2 * dt  # beta_s2
        state[5] = 0  # beta_c3 (направляющее колесо)
        
        states[i] = state
    
    return states

def create_reference_trajectory_type12(x_traj, y_traj, theta_traj, beta_s1_traj, beta_s2_traj, t):
    """
    Создание функции эталонной траектории для робота типа (1,2)
    """
    from scipy.interpolate import interp1d
    
    # Интерполяция позиций
    x_ref_func = interp1d(t, x_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    y_ref_func = interp1d(t, y_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    theta_ref_func = interp1d(t, theta_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    beta_s1_ref_func = interp1d(t, beta_s1_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    beta_s2_ref_func = interp1d(t, beta_s2_traj, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    # Вычисление скоростей
    eta1_ref = np.gradient(x_traj, t)
    eta2_ref = np.gradient(y_traj, t)
    zeta1_ref = np.gradient(beta_s1_traj, t)
    zeta2_ref = np.gradient(beta_s2_traj, t)
    
    eta1_ref_func = interp1d(t, eta1_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    eta2_ref_func = interp1d(t, eta2_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    zeta1_ref_func = interp1d(t, zeta1_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    zeta2_ref_func = interp1d(t, zeta2_ref, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    def ref_trajectory(t):
        ref_state = np.array([x_ref_func(t), y_ref_func(t), theta_ref_func(t)])
        ref_eta = np.array([eta1_ref_func(t), eta2_ref_func(t)])
        ref_zeta = np.array([zeta1_ref_func(t), zeta2_ref_func(t)])
        return ref_state, ref_eta, ref_zeta
    
    return ref_trajectory

def plot_comparison_type12(t, x_ref, y_ref, theta_ref, beta_s1_ref, beta_s2_ref,
                          x_static, y_static, theta_static, beta_s1_static, beta_s2_static,
                          x_dynamic, y_dynamic, theta_dynamic, beta_s1_dynamic, beta_s2_dynamic,
                          title="Сравнение методов линеаризации для робота типа (1,2) (вариант 5)"):
    """Построение графиков сравнения результатов для робота типа (1,2)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
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
    
    # Углы поворота передних колес
    ax3.plot(t, beta_s1_ref, 'k--', linewidth=2, label='Эталонный βs1')
    ax3.plot(t, beta_s1_static, 'b-', linewidth=2, label='βs1 (статическая)')
    ax3.plot(t, beta_s1_dynamic, 'r-', linewidth=2, label='βs1 (динамическая)')
    ax3.plot(t, beta_s2_ref, 'k:', linewidth=2, label='Эталонный βs2')
    ax3.plot(t, beta_s2_static, 'b--', linewidth=2, label='βs2 (статическая)')
    ax3.plot(t, beta_s2_dynamic, 'r--', linewidth=2, label='βs2 (динамическая)')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Угол поворота колеса (рад)')
    ax3.set_title('Углы поворота передних колес')
    ax3.legend()
    ax3.grid(True)
    
    # Ошибки углов поворота
    e_beta_s1_static = beta_s1_ref - beta_s1_static
    e_beta_s2_static = beta_s2_ref - beta_s2_static
    e_beta_s1_dynamic = beta_s1_ref - beta_s1_dynamic
    e_beta_s2_dynamic = beta_s2_ref - beta_s2_dynamic
    
    ax4.plot(t, e_beta_s1_static, 'b-', linewidth=2, label='Ошибка βs1 (статическая)')
    ax4.plot(t, e_beta_s2_static, 'b--', linewidth=2, label='Ошибка βs2 (статическая)')
    ax4.plot(t, e_beta_s1_dynamic, 'r-', linewidth=2, label='Ошибка βs1 (динамическая)')
    ax4.plot(t, e_beta_s2_dynamic, 'r--', linewidth=2, label='Ошибка βs2 (динамическая)')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Ошибка угла поворота (рад)')
    ax4.set_title('Ошибки углов поворота колес')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    from robot_model_type12 import generate_trajectory_variant5_type12
    
    # Временной интервал
    t = np.linspace(0, 30, 1500)
    
    # Генерация эталонной траектории для варианта 5
    x_ref, y_ref, theta_ref, beta_s1_ref, beta_s2_ref = generate_trajectory_variant5_type12(
        t, R1=7.0, R2=12.0, alpha=np.pi/6, delta=2*np.pi, t_straight=6.0
    )
    
    # Создание функции эталонной траектории
    ref_trajectory = create_reference_trajectory_type12(x_ref, y_ref, theta_ref, beta_s1_ref, beta_s2_ref, t)
    
    # Начальное состояние для варианта 5: [0, 3, 2π/3, 0, 0, 0]
    x0 = np.array([0.1, 3.1, 2*np.pi/3 + 0.05, 0.0, 0.0, 0.0])
    
    # Симуляция статической линеаризации
    states_static = simulate_robot_tracking_type12(x0, t, ref_trajectory, 'static', 
                                                  kp1=1.0, kp2=1.0, kp_beta=0.5)
    
    # Симуляция динамической линеаризации
    states_dynamic = simulate_robot_tracking_type12(x0, t, ref_trajectory, 'dynamic', 
                                                   kp1=1.0, kp2=1.0, kp_beta=0.5,
                                                   kd1=0.5, kd2=0.5, kd_beta=0.2,
                                                   ki1=0.1, ki2=0.1, ki_beta=0.05)
    
    # Извлечение результатов
    x_static = states_static[:, 0]
    y_static = states_static[:, 1]
    theta_static = states_static[:, 2]
    beta_s1_static = states_static[:, 3]
    beta_s2_static = states_static[:, 4]
    
    x_dynamic = states_dynamic[:, 0]
    y_dynamic = states_dynamic[:, 1]
    theta_dynamic = states_dynamic[:, 2]
    beta_s1_dynamic = states_dynamic[:, 3]
    beta_s2_dynamic = states_dynamic[:, 4]
    
    # Построение графиков
    fig = plot_comparison_type12(t, x_ref, y_ref, theta_ref, beta_s1_ref, beta_s2_ref,
                                x_static, y_static, theta_static, beta_s1_static, beta_s2_static,
                                x_dynamic, y_dynamic, theta_dynamic, beta_s1_dynamic, beta_s2_dynamic)
    
    # Сохранение результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab2/images/task2"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, "controllers_comparison_type12.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение данных
    np.savez(os.path.join(output_dir, "controllers_type12_results.npz"),
             t=t, x_ref=x_ref, y_ref=y_ref, theta_ref=theta_ref, 
             beta_s1_ref=beta_s1_ref, beta_s2_ref=beta_s2_ref,
             x_static=x_static, y_static=y_static, theta_static=theta_static,
             beta_s1_static=beta_s1_static, beta_s2_static=beta_s2_static,
             x_dynamic=x_dynamic, y_dynamic=y_dynamic, theta_dynamic=theta_dynamic,
             beta_s1_dynamic=beta_s1_dynamic, beta_s2_dynamic=beta_s2_dynamic)
    
    # Вычисление ошибок
    max_error_x_static = np.max(np.abs(x_ref - x_static))
    max_error_y_static = np.max(np.abs(y_ref - y_static))
    max_error_theta_static = np.max(np.abs(theta_ref - theta_static))
    max_error_beta_s1_static = np.max(np.abs(beta_s1_ref - beta_s1_static))
    max_error_beta_s2_static = np.max(np.abs(beta_s2_ref - beta_s2_static))
    
    max_error_x_dynamic = np.max(np.abs(x_ref - x_dynamic))
    max_error_y_dynamic = np.max(np.abs(y_ref - y_dynamic))
    max_error_theta_dynamic = np.max(np.abs(theta_ref - theta_dynamic))
    max_error_beta_s1_dynamic = np.max(np.abs(beta_s1_ref - beta_s1_dynamic))
    max_error_beta_s2_dynamic = np.max(np.abs(beta_s2_ref - beta_s2_dynamic))
    
    print(f"Результаты сравнения контроллеров для робота типа (1,2) сохранены в {output_dir}")
    print("\nСтатическая линеаризация:")
    print(f"  Максимальная ошибка по X: {max_error_x_static:.4f} м")
    print(f"  Максимальная ошибка по Y: {max_error_y_static:.4f} м")
    print(f"  Максимальная ошибка по углу: {max_error_theta_static:.4f} рад")
    print(f"  Максимальная ошибка по βs1: {max_error_beta_s1_static:.4f} рад")
    print(f"  Максимальная ошибка по βs2: {max_error_beta_s2_static:.4f} рад")
    
    print("\nДинамическая линеаризация:")
    print(f"  Максимальная ошибка по X: {max_error_x_dynamic:.4f} м")
    print(f"  Максимальная ошибка по Y: {max_error_y_dynamic:.4f} м")
    print(f"  Максимальная ошибка по углу: {max_error_theta_dynamic:.4f} рад")
    print(f"  Максимальная ошибка по βs1: {max_error_beta_s1_dynamic:.4f} рад")
    print(f"  Максимальная ошибка по βs2: {max_error_beta_s2_dynamic:.4f} рад")

if __name__ == "__main__":
    main()
