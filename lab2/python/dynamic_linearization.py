import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class DynamicFeedbackLinearization:
    """
    Динамическая линеаризация обратной связи для мобильного робота
    """
    
    def __init__(self, robot, kp=2.0, kd=1.0, ki=0.1):
        self.robot = robot
        self.kp = kp  # коэффициент пропорционального управления
        self.kd = kd  # коэффициент дифференциального управления
        self.ki = ki  # коэффициент интегрального управления
        
        # Интегральные ошибки
        self.integral_ex = 0.0
        self.integral_ey = 0.0
        self.integral_etheta = 0.0
        
    def control_law(self, state, ref_state, ref_vel, ref_acc, dt):
        """
        Закон управления с динамической линеаризацией
        state = [x, y, theta, vx, vy, omega] - текущее состояние
        ref_state = [x_ref, y_ref, theta_ref] - желаемое состояние
        ref_vel = [vx_ref, vy_ref, omega_ref] - желаемая скорость
        ref_acc = [ax_ref, ay_ref, alpha_ref] - желаемое ускорение
        """
        x, y, theta, vx, vy, omega = state
        x_ref, y_ref, theta_ref = ref_state
        vx_ref, vy_ref, omega_ref = ref_vel
        ax_ref, ay_ref, alpha_ref = ref_acc
        
        # Ошибки позиции
        ex = x_ref - x
        ey = y_ref - y
        etheta = theta_ref - theta
        
        # Ошибки скорости
        evx = vx_ref - vx
        evy = vy_ref - vy
        eomega = omega_ref - omega
        
        # Обновление интегральных ошибок
        self.integral_ex += ex * dt
        self.integral_ey += ey * dt
        self.integral_etheta += etheta * dt
        
        # Динамическая линеаризация с учетом ускорений
        u1 = ax_ref + self.kp * ex + self.kd * evx + self.ki * self.integral_ex
        u2 = ay_ref + self.kp * ey + self.kd * evy + self.ki * self.integral_ey
        u3 = alpha_ref + self.kp * etheta + self.kd * eomega + self.ki * self.integral_etheta
        
        # Преобразование в управляющие сигналы колес
        u_wheels = self.robot.R * np.array([u1, u2, u3])
        
        return u_wheels
    
    def simulate_tracking(self, x0, t_span, ref_trajectory):
        """
        Симуляция слежения за траекторией с динамической линеаризацией
        """
        def rhs(state, t):
            # Получение эталонной траектории в момент времени t
            ref_state, ref_vel, ref_acc = ref_trajectory(t)
            
            # Вычисление шага времени
            if hasattr(rhs, 'prev_t'):
                dt = t - rhs.prev_t
            else:
                dt = t_span[1] - t_span[0] if len(t_span) > 1 else 0.01
            rhs.prev_t = t
            
            # Вычисление управления
            u = self.control_law(state, ref_state, ref_vel, ref_acc, dt)
            
            # Динамика робота
            return self.robot.dynamics(state, t, u)
        
        sol = odeint(rhs, x0, t_span)
        return sol

def create_reference_trajectory_with_acceleration(x_traj, y_traj, theta_traj, t):
    """
    Создание функции эталонной траектории с ускорениями
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

def plot_dynamic_tracking_results(t, x_ref, y_ref, theta_ref, x_actual, y_actual, theta_actual, 
                                 title="Результаты динамической линеаризации"):
    """Построение графиков результатов динамического слежения"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Траектория в плоскости
    ax1.plot(x_ref, y_ref, 'b--', linewidth=2, label='Эталонная траектория')
    ax1.plot(x_actual, y_actual, 'r-', linewidth=2, label='Фактическая траектория')
    ax1.plot(x_ref[0], y_ref[0], 'go', markersize=8, label='Начало')
    ax1.plot(x_ref[-1], y_ref[-1], 'ro', markersize=8, label='Конец')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектория в плоскости')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Ошибки позиции
    ex = x_ref - x_actual
    ey = y_ref - y_actual
    ax2.plot(t, ex, 'b-', linewidth=2, label='Ошибка по X')
    ax2.plot(t, ey, 'r-', linewidth=2, label='Ошибка по Y')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Ошибка позиции (м)')
    ax2.set_title('Ошибки позиции')
    ax2.legend()
    ax2.grid(True)
    
    # Угол поворота
    ax3.plot(t, theta_ref, 'b--', linewidth=2, label='Эталонный угол')
    ax3.plot(t, theta_actual, 'r-', linewidth=2, label='Фактический угол')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Угол поворота (рад)')
    ax3.set_title('Ориентация робота')
    ax3.legend()
    ax3.grid(True)
    
    # Ошибка угла
    etheta = theta_ref - theta_actual
    ax4.plot(t, etheta, 'g-', linewidth=2, label='Ошибка угла')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Ошибка угла (рад)')
    ax4.set_title('Ошибка ориентации')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    from robot_model import FourWheelMobileRobot, generate_trajectory
    
    # Создание робота
    robot = FourWheelMobileRobot(L=0.3, W=0.2, R=0.05, m=10.0, I=1.0)
    
    # Временной интервал
    t = np.linspace(0, 10, 1000)
    
    # Генерация эталонной траектории
    x_ref, y_ref, theta_ref = generate_trajectory(t, R1=2.0, R2=1.0, 
                                                alpha=np.pi/4, delta=np.pi/2, 
                                                t_straight=3.0)
    
    # Создание функции эталонной траектории с ускорениями
    ref_trajectory = create_reference_trajectory_with_acceleration(x_ref, y_ref, theta_ref, t)
    
    # Создание контроллера
    controller = DynamicFeedbackLinearization(robot, kp=2.0, kd=1.0, ki=0.1)
    
    # Начальное состояние (с небольшим отклонением от эталонной траектории)
    x0 = np.array([0.1, 0.1, 0.05, 0.0, 0.0, 0.0])
    
    # Симуляция слежения
    sol = controller.simulate_tracking(x0, t, ref_trajectory)
    
    # Извлечение результатов
    x_actual = sol[:, 0]
    y_actual = sol[:, 1]
    theta_actual = sol[:, 2]
    
    # Построение графиков
    fig = plot_dynamic_tracking_results(t, x_ref, y_ref, theta_ref, 
                                      x_actual, y_actual, theta_actual,
                                      "Динамическая линеаризация обратной связи")
    
    # Сохранение результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab2/images/task4"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, "dynamic_linearization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение данных
    np.savez(os.path.join(output_dir, "dynamic_results.npz"),
             t=t, x_ref=x_ref, y_ref=y_ref, theta_ref=theta_ref,
             x_actual=x_actual, y_actual=y_actual, theta_actual=theta_actual)
    
    # Вычисление ошибок
    max_error_x = np.max(np.abs(x_ref - x_actual))
    max_error_y = np.max(np.abs(y_ref - y_actual))
    max_error_theta = np.max(np.abs(theta_ref - theta_actual))
    
    print(f"Результаты динамической линеаризации сохранены в {output_dir}")
    print(f"Максимальная ошибка по X: {max_error_x:.4f} м")
    print(f"Максимальная ошибка по Y: {max_error_y:.4f} м")
    print(f"Максимальная ошибка по углу: {max_error_theta:.4f} рад")

if __name__ == "__main__":
    main()
