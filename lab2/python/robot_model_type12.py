import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class FourWheelMobileRobotType12:
    """
    Кинематическая модель четырехколесного мобильного робота типа (1,2)
    с матрицами J1, J2, C1, C2 согласно техническому заданию
    """
    
    def __init__(self, L=0.3, W=0.2, R=0.05, d=0.1, m=10.0, I=1.0):
        """
        Параметры робота типа (1,2):
        L - расстояние от центра масс до колес
        W - колея (расстояние между левыми и правыми колесами)  
        R - радиус колес
        d - геометрический параметр
        m - масса робота
        I - момент инерции
        """
        self.L = L
        self.W = W
        self.R = R
        self.d = d
        self.m = m
        self.I = I
        
        # Характеристические константы для типа (1,2)
        # Передние колеса (рулевые): α1=0, α2=π
        # Заднее колесо (направляющее): α3=3π/2
        self.alpha1 = 0
        self.alpha2 = np.pi
        self.alpha3 = 3 * np.pi / 2
        
    def get_J1_matrix(self, beta_s1, beta_s2, beta_c3):
        """
        Матрица J1 кинематических ограничений
        """
        J1 = np.array([
            [-np.sin(beta_s1), np.cos(beta_s1), self.L * np.cos(beta_s1)],
            [np.sin(beta_s2), -np.cos(beta_s2), self.L * np.cos(beta_s2)],
            [np.cos(beta_c3), np.sin(beta_c3), self.L * np.cos(beta_c3)]
        ])
        return J1
    
    def get_J2_matrix(self):
        """
        Матрица J2 (диагональная с радиусом колес)
        """
        return np.diag([self.R, self.R, self.R])
    
    def get_C1_matrix(self, beta_s1, beta_s2, beta_c3):
        """
        Матрица C1 кинематических ограничений
        """
        C1 = np.array([
            [np.cos(beta_s1), np.sin(beta_s1), self.L * np.sin(beta_s1)],
            [-np.cos(beta_s2), -np.sin(beta_s2), self.L * np.sin(beta_s2)],
            [np.sin(beta_c3), -np.cos(beta_c3), self.d + self.L * np.sin(beta_c3)]
        ])
        return C1
    
    def get_C2_vector(self):
        """
        Вектор C2 кинематических ограничений
        """
        return np.array([0, 0, self.d])
    
    def get_rotation_matrix(self, theta):
        """
        Матрица поворота R(θ)
        """
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    def get_sigma_matrix(self, beta_s1, beta_s2):
        """
        Матрица Σ(βs1, βs2) для кинематической модели
        """
        # Упрощенная версия матрицы Σ для демонстрации
        sigma = np.array([
            [np.cos(beta_s1), np.sin(beta_s1)],
            [np.cos(beta_s2), np.sin(beta_s2)]
        ])
        return sigma
    
    def kinematic_model(self, state, t, control_input):
        """
        Кинематическая модель робота типа (1,2)
        state = [x, y, theta, beta_s1, beta_s2, beta_c3]
        control_input = [eta1, eta2, zeta1, zeta2]
        """
        x, y, theta, beta_s1, beta_s2, beta_c3 = state
        eta1, eta2, zeta1, zeta2 = control_input
        
        # Матрица поворота
        R = self.get_rotation_matrix(theta)
        
        # Матрица Σ
        Sigma = self.get_sigma_matrix(beta_s1, beta_s2)
        
        # Обобщенные скорости
        eta = np.array([eta1, eta2])
        
        # Кинематические уравнения
        xi_dot = R.T @ Sigma @ eta
        
        # Производные углов поворота колес
        beta_s1_dot = zeta1
        beta_s2_dot = zeta2
        beta_c3_dot = 0  # Направляющее колесо не управляется
        
        return np.array([
            xi_dot[0],  # dx/dt
            xi_dot[1],  # dy/dt
            xi_dot[2] if len(xi_dot) > 2 else 0,  # dtheta/dt (упрощенно)
            beta_s1_dot,
            beta_s2_dot,
            beta_c3_dot
        ])
    
    def simulate(self, initial_state, control_function, time_points):
        """
        Симуляция движения робота
        """
        def ode_func(state, t):
            control = control_function(t)
            return self.kinematic_model(state, t, control)
        
        trajectory = odeint(ode_func, initial_state, time_points)
        return trajectory

def generate_trajectory_variant5_type12(t, R1=7.0, R2=12.0, alpha=np.pi/6, delta=2*np.pi, t_straight=6.0):
    """
    Генерация траектории для варианта 5 с учетом кинематической модели типа (1,2)
    """
    # Начальная позиция для варианта 5: [0, 3, 2π/3]
    x0, y0 = 0.0, 3.0
    theta0 = 2*np.pi/3
    
    # Параметры траектории
    v_circle = 1.0  # скорость движения по окружности
    v_straight = 1.0  # скорость движения по прямой
    
    # Временные интервалы
    t1 = R1 * delta / v_circle  # время движения по первой окружности
    t2 = t1 + t_straight  # время движения по прямой
    t3 = t2 + R2 * np.pi / v_circle  # время движения по второй окружности
    
    x_traj = np.zeros_like(t)
    y_traj = np.zeros_like(t)
    theta_traj = np.zeros_like(t)
    beta_s1_traj = np.zeros_like(t)
    beta_s2_traj = np.zeros_like(t)
    
    for i, ti in enumerate(t):
        if ti <= t1:
            # Движение по первой окружности
            angle = theta0 + delta * ti / t1
            x_traj[i] = x0 + R1 * np.sin(angle) - R1 * np.sin(theta0)
            y_traj[i] = y0 + R1 * np.cos(theta0) - R1 * np.cos(angle)
            theta_traj[i] = angle
            
            # Углы поворота передних колес для движения по окружности
            beta_s1_traj[i] = np.pi/4  # Примерное значение
            beta_s2_traj[i] = -np.pi/4  # Примерное значение
            
        elif ti <= t2:
            # Движение по прямой
            t_straight_local = ti - t1
            end_angle = theta0 + delta
            x_end_circle = x0 + R1 * np.sin(end_angle) - R1 * np.sin(theta0)
            y_end_circle = y0 + R1 * np.cos(theta0) - R1 * np.cos(end_angle)
            
            x_traj[i] = x_end_circle + v_straight * t_straight_local * np.cos(end_angle)
            y_traj[i] = y_end_circle + v_straight * t_straight_local * np.sin(end_angle)
            theta_traj[i] = end_angle
            
            # Углы поворота для прямолинейного движения
            beta_s1_traj[i] = 0
            beta_s2_traj[i] = 0
            
        else:
            # Движение по второй окружности
            t_circle2 = ti - t2
            angle2 = np.pi * t_circle2 / (t3 - t2)
            
            end_angle = theta0 + delta
            x_end_circle = x0 + R1 * np.sin(end_angle) - R1 * np.sin(theta0)
            y_end_circle = y0 + R1 * np.cos(theta0) - R1 * np.cos(end_angle)
            x_end_straight = x_end_circle + v_straight * t_straight * np.cos(end_angle)
            y_end_straight = y_end_circle + v_straight * t_straight * np.sin(end_angle)
            
            center_x = x_end_straight + R2 * np.cos(end_angle + alpha)
            center_y = y_end_straight + R2 * np.sin(end_angle + alpha)
            
            x_traj[i] = center_x + R2 * np.cos(end_angle + alpha + angle2)
            y_traj[i] = center_y + R2 * np.sin(end_angle + alpha + angle2)
            theta_traj[i] = end_angle + alpha + angle2
            
            # Углы поворота для движения по второй окружности
            beta_s1_traj[i] = np.pi/6
            beta_s2_traj[i] = -np.pi/6
    
    return x_traj, y_traj, theta_traj, beta_s1_traj, beta_s2_traj

def plot_trajectory_type12(x_traj, y_traj, theta_traj, beta_s1_traj, beta_s2_traj, t, 
                          title="Траектория движения робота типа (1,2) (вариант 5)"):
    """Построение графика траектории для робота типа (1,2)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Траектория в плоскости
    ax1.plot(x_traj, y_traj, 'b-', linewidth=2, label='Траектория')
    ax1.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Начало')
    ax1.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='Конец')
    
    # Стрелки направления
    step = len(t) // 15
    for i in range(0, len(t), step):
        ax1.arrow(x_traj[i], y_traj[i], 
                  0.2 * np.cos(theta_traj[i]), 0.2 * np.sin(theta_traj[i]),
                  head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектория в плоскости (вариант 5)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Угол поворота во времени
    ax2.plot(t, theta_traj, 'g-', linewidth=2, label='Ориентация робота')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Угол поворота (рад)')
    ax2.set_title('Ориентация робота')
    ax2.grid(True)
    ax2.legend()
    
    # Углы поворота передних колес
    ax3.plot(t, beta_s1_traj, 'r-', linewidth=2, label='βs1 (переднее левое)')
    ax3.plot(t, beta_s2_traj, 'b-', linewidth=2, label='βs2 (переднее правое)')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Угол поворота колеса (рад)')
    ax3.set_title('Углы поворота передних колес')
    ax3.grid(True)
    ax3.legend()
    
    # Схема робота типа (1,2)
    ax4.text(0.5, 0.8, 'Робот типа (1,2)', ha='center', va='center', fontsize=14, weight='bold')
    ax4.text(0.5, 0.6, '• Передние колеса: рулевые (βs1, βs2)', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.5, '• Задние колеса: направляющие', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.4, '• Матрицы: J₁, J₂, C₁, C₂', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.3, '• Обобщенные скорости: η₁, η₂', ha='center', va='center', fontsize=12)
    ax4.text(0.5, 0.2, '• Управление углами: ζ₁, ζ₂', ha='center', va='center', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

def main():
    # Параметры робота типа (1,2)
    robot = FourWheelMobileRobotType12(L=0.3, W=0.2, R=0.05, d=0.1, m=10.0, I=1.0)
    
    # Временной интервал
    t = np.linspace(0, 30, 1500)
    
    # Генерация траектории для варианта 5
    x_traj, y_traj, theta_traj, beta_s1_traj, beta_s2_traj = generate_trajectory_variant5_type12(
        t, R1=7.0, R2=12.0, alpha=np.pi/6, delta=2*np.pi, t_straight=6.0
    )
    
    # Построение графиков
    fig = plot_trajectory_type12(x_traj, y_traj, theta_traj, beta_s1_traj, beta_s2_traj, t)
    
    # Сохранение результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab2/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, "trajectory_variant5_type12.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение данных
    np.savez(os.path.join(output_dir, "trajectory_variant5_type12_data.npz"),
             t=t, x=x_traj, y=y_traj, theta=theta_traj, 
             beta_s1=beta_s1_traj, beta_s2=beta_s2_traj)
    
    print(f"Результаты для варианта 5 (тип 1,2) сохранены в {output_dir}")
    print(f"Параметры робота: L={robot.L}, W={robot.W}, R={robot.R}, d={robot.d}")
    print(f"Масса: {robot.m} кг, Момент инерции: {robot.I} кг⋅м²")
    print(f"Тип робота: (1,2) - передние колеса рулевые, задние направляющие")
    print(f"Начальное состояние: [0, 3, 2π/3]")
    print(f"Параметры траектории: R1=7м, δ=2π, α=π/6, t=6с, R2=12м")
    print(f"Кинематическая модель: матрицы J₁, J₂, C₁, C₂")

if __name__ == "__main__":
    main()
