import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class FourWheelMobileRobotVariant5:
    """
    Модель четырехколесного мобильного робота с дифференциальным приводом для варианта 5
    """
    
    def __init__(self, L=0.3, W=0.2, R=0.05, m=10.0, I=1.0):
        """
        Параметры робота для варианта 5:
        L - база робота (расстояние между передними и задними колесами)
        W - колея (расстояние между левыми и правыми колесами)  
        R - радиус колес
        m - масса робота
        I - момент инерции
        """
        self.L = L
        self.W = W
        self.R = R
        self.m = m
        self.I = I
        
        # Матрица конфигурации для дифференциального привода
        # v = (v_L + v_R) / 2, omega = (v_R - v_L) / L
        self.B = np.array([
            [1, 0],  # v - линейная скорость
            [0, 1]   # omega - угловая скорость
        ])
    
    def dynamics(self, state, t, control_input):
        """
        Динамика четырехколесного робота с дифференциальным приводом
        state = [x, y, theta] - позиция и ориентация
        control_input = [v, omega] - линейная и угловая скорости
        """
        x, y, theta = state
        v, omega = control_input
        
        # Динамические уравнения для дифференциального привода
        dx_dt = v * np.cos(theta)
        dy_dt = v * np.sin(theta)
        dtheta_dt = omega
        
        return np.array([dx_dt, dy_dt, dtheta_dt])
    
    def simulate_trajectory(self, initial_state, control_sequence, time_points):
        """
        Симуляция траектории робота
        """
        def ode_func(state, t):
            # Интерполяция управления по времени
            if len(control_sequence) == 1:
                control = control_sequence[0]
            else:
                # Простая интерполяция для демонстрации
                control = control_sequence[0]
            
            return self.dynamics(state, t, control)
        
        trajectory = odeint(ode_func, initial_state, time_points)
        return trajectory
    
    def generate_trajectory_variant5(self):
        """
        Генерация траектории для варианта 5
        """
        # Параметры траектории из задания
        R1 = 7.0
        delta = 2 * np.pi
        alpha = np.pi / 6
        t_straight = 6.0
        
        # Начальное состояние
        initial_state = np.array([0, 3, 2 * np.pi / 3])
        
        # Участок 1: Движение по окружности R1
        t1_duration = R1 * delta / 2.0  # Примерная скорость
        t1 = np.linspace(0, t1_duration, 500)
        
        # Центр первой окружности
        center_x = initial_state[0] - R1 * np.sin(initial_state[2])
        center_y = initial_state[1] + R1 * np.cos(initial_state[2])
        
        x1 = center_x + R1 * np.sin(initial_state[2] + (delta * t1 / t1_duration))
        y1 = center_y - R1 * np.cos(initial_state[2] + (delta * t1 / t1_duration))
        theta1 = initial_state[2] + (delta * t1 / t1_duration)
        
        end_state1 = np.array([x1[-1], y1[-1], theta1[-1]])
        
        # Участок 2: Движение по прямой
        t2 = np.linspace(0, t_straight, 200)
        v_straight = 2.0
        x2 = end_state1[0] + v_straight * t2 * np.cos(end_state1[2] + alpha)
        y2 = end_state1[1] + v_straight * t2 * np.sin(end_state1[2] + alpha)
        theta2 = np.full_like(t2, end_state1[2] + alpha)
        
        end_state2 = np.array([x2[-1], y2[-1], theta2[-1]])
        
        # Участок 3: Движение по окружности R2
        R2 = 12.0
        omega_circle2 = -2.0 / R2  # По часовой стрелке
        t3_duration = R2 * np.pi / 2.0
        t3 = np.linspace(0, t3_duration, 500)
        
        # Центр второй окружности
        center_x2 = end_state2[0] - R2 * np.sin(end_state2[2])
        center_y2 = end_state2[1] + R2 * np.cos(end_state2[2])
        
        x3 = center_x2 + R2 * np.sin(end_state2[2] + omega_circle2 * t3)
        y3 = center_y2 - R2 * np.cos(end_state2[2] + omega_circle2 * t3)
        theta3 = end_state2[2] + omega_circle2 * t3
        
        # Объединяем траектории
        x_full = np.concatenate((x1, x2, x3))
        y_full = np.concatenate((y1, y2, y3))
        theta_full = np.concatenate((theta1, theta2, theta3))
        
        return {
            'x': x_full,
            'y': y_full,
            'theta': theta_full,
            'segments': {
                'segment1': {'x': x1, 'y': y1, 'theta': theta1, 't': t1},
                'segment2': {'x': x2, 'y': y2, 'theta': theta2, 't': t2},
                'segment3': {'x': x3, 'y': y3, 'theta': theta3, 't': t3}
            }
        }
    
    def plot_trajectory(self, trajectory_data, save_path=None):
        """
        Построение графика траектории
        """
        plt.figure(figsize=(10, 8))
        
        # Участки траектории
        seg1 = trajectory_data['segments']['segment1']
        seg2 = trajectory_data['segments']['segment2']
        seg3 = trajectory_data['segments']['segment3']
        
        plt.plot(seg1['x'], seg1['y'], 'b-', linewidth=2, label='Участок 1 (Окружность R1=7м)')
        plt.plot(seg2['x'], seg2['y'], 'g-', linewidth=2, label='Участок 2 (Прямая t=6с)')
        plt.plot(seg3['x'], seg3['y'], 'r-', linewidth=2, label='Участок 3 (Окружность R2=12м)')
        
        # Начальная точка
        plt.plot(0, 3, 'go', markersize=8, label='Начальная позиция')
        
        # Переходные точки
        plt.plot(seg1['x'][-1], seg1['y'][-1], 'bs', markersize=8, label='Переход 1')
        plt.plot(seg2['x'][-1], seg2['y'][-1], 'gs', markersize=8, label='Переход 2')
        
        # Стрелки для направления
        x_full = trajectory_data['x']
        y_full = trajectory_data['y']
        theta_full = trajectory_data['theta']
        
        for i in range(0, len(x_full), 50):
            plt.arrow(x_full[i], y_full[i], 0.05 * np.cos(theta_full[i]), 0.05 * np.sin(theta_full[i]),
                      head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
        
        plt.title('Траектория движения четырехколесного робота с дифференциальным приводом (Вариант 5)')
        plt.xlabel('X, м')
        plt.ylabel('Y, м')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    """
    Основная функция для демонстрации работы модели
    """
    # Создание модели робота
    robot = FourWheelMobileRobotVariant5()
    
    # Генерация траектории
    trajectory_data = robot.generate_trajectory_variant5()
    
    # Построение графика
    output_dir = "images/task1"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "trajectory_variant5_detailed.png")
    
    robot.plot_trajectory(trajectory_data, output_file)
    print(f"Траектория сохранена: {output_file}")
    
    # Сохранение данных траектории
    np.savez(os.path.join(output_dir, "trajectory_variant5_data.npz"), **trajectory_data)
    print(f"Данные траектории сохранены: {os.path.join(output_dir, 'trajectory_variant5_data.npz')}")

if __name__ == "__main__":
    main()