import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class FourWheelMobileRobotVariant5:
    """
    Модель четырехколесного мобильного робота типа (1,2) для варианта 5
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
        
        # Матрица конфигурации приводов для типа (1,2)
        # Тип (1,2) означает: 1 - передние колеса управляемые, 2 - задние колеса управляемые
        self.B = np.array([
            [1, 1, 0, 0],  # vx - только передние колеса
            [0, 0, 1, 1],  # vy - только задние колеса
            [1, -1, 1, -1] # omega - все колеса для поворота
        ]) / (2 * R)
        
    def dynamics(self, state, t, u):
        """
        Динамическая модель робота
        state = [x, y, theta, vx, vy, omega]
        u = [u1, u2, u3, u4] - управляющие сигналы на колеса
        """
        x, y, theta, vx, vy, omega = state
        
        # Преобразование управляющих сигналов в скорости
        if len(u) == 3:
            # Если передано 3 управляющих сигнала, расширяем до 4
            u = np.append(u, 0.0)
        elif len(u) != 4:
            raise ValueError("Управляющий сигнал должен быть 3- или 4-мерным")
            
        v = self.B @ u
        
        # Динамические уравнения
        dx_dt = vx * np.cos(theta) - vy * np.sin(theta)
        dy_dt = vx * np.sin(theta) + vy * np.cos(theta)
        dtheta_dt = omega
        
        # Упрощенная динамика (без учета сил трения и инерции)
        dvx_dt = v[0]
        dvy_dt = v[1] 
        domega_dt = v[2]
        
        return [dx_dt, dy_dt, dtheta_dt, dvx_dt, dvy_dt, domega_dt]
    
    def simulate(self, x0, t_span, u_func):
        """
        Симуляция движения робота
        x0 - начальное состояние
        t_span - временной интервал
        u_func - функция управления u(t)
        """
        def rhs(state, t):
            u = u_func(t)
            return self.dynamics(state, t, u)
            
        sol = odeint(rhs, x0, t_span)
        return sol

def generate_trajectory_variant5(t, R1=7.0, R2=12.0, alpha=np.pi/6, delta=2*np.pi, t_straight=6.0):
    """
    Генерация траектории для варианта 5
    R1 = 7 м - радиус первой окружности
    R2 = 12 м - радиус второй окружности  
    alpha = π/6 - угол поворота
    delta = 2π - угол дуги первой окружности
    t_straight = 6 сек - время движения по прямой
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
    
    for i, ti in enumerate(t):
        if ti <= t1:
            # Движение по первой окружности (положительное направление)
            angle = theta0 + delta * ti / t1
            x_traj[i] = x0 + R1 * np.sin(angle) - R1 * np.sin(theta0)
            y_traj[i] = y0 + R1 * np.cos(theta0) - R1 * np.cos(angle)
            theta_traj[i] = angle
        elif ti <= t2:
            # Движение по прямой
            t_straight_local = ti - t1
            # Конечная точка первой окружности
            end_angle = theta0 + delta
            x_end_circle = x0 + R1 * np.sin(end_angle) - R1 * np.sin(theta0)
            y_end_circle = y0 + R1 * np.cos(theta0) - R1 * np.cos(end_angle)
            
            x_traj[i] = x_end_circle + v_straight * t_straight_local * np.cos(end_angle)
            y_traj[i] = y_end_circle + v_straight * t_straight_local * np.sin(end_angle)
            theta_traj[i] = end_angle
        else:
            # Движение по второй окружности (по часовой стрелке)
            t_circle2 = ti - t2
            angle2 = np.pi * t_circle2 / (t3 - t2)
            
            # Конечная точка прямой
            end_angle = theta0 + delta
            x_end_circle = x0 + R1 * np.sin(end_angle) - R1 * np.sin(theta0)
            y_end_circle = y0 + R1 * np.cos(theta0) - R1 * np.cos(end_angle)
            x_end_straight = x_end_circle + v_straight * t_straight * np.cos(end_angle)
            y_end_straight = y_end_circle + v_straight * t_straight * np.sin(end_angle)
            
            # Центр второй окружности
            center_x = x_end_straight + R2 * np.cos(end_angle + alpha)
            center_y = y_end_straight + R2 * np.sin(end_angle + alpha)
            
            # Движение по второй окружности (по часовой стрелке)
            x_traj[i] = center_x + R2 * np.cos(end_angle + alpha + angle2)
            y_traj[i] = center_y + R2 * np.sin(end_angle + alpha + angle2)
            theta_traj[i] = end_angle + alpha + angle2
    
    return x_traj, y_traj, theta_traj

def plot_trajectory_variant5(x_traj, y_traj, theta_traj, t, title="Траектория движения робота (вариант 5)"):
    """Построение графика траектории для варианта 5"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    ax2.plot(t, theta_traj, 'g-', linewidth=2)
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Угол поворота (рад)')
    ax2.set_title('Ориентация робота')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # Параметры робота для варианта 5
    robot = FourWheelMobileRobotVariant5(L=0.3, W=0.2, R=0.05, m=10.0, I=1.0)
    
    # Временной интервал (увеличиваем для полной траектории)
    t = np.linspace(0, 30, 1500)
    
    # Генерация траектории для варианта 5
    x_traj, y_traj, theta_traj = generate_trajectory_variant5(t, R1=7.0, R2=12.0, 
                                                             alpha=np.pi/6, delta=2*np.pi, 
                                                             t_straight=6.0)
    
    # Построение графиков
    fig = plot_trajectory_variant5(x_traj, y_traj, theta_traj, t)
    
    # Сохранение результатов
    output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab2/images/task1"
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(os.path.join(output_dir, "trajectory_variant5.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Сохранение данных
    np.savez(os.path.join(output_dir, "trajectory_variant5_data.npz"),
             t=t, x=x_traj, y=y_traj, theta=theta_traj)
    
    print(f"Результаты для варианта 5 сохранены в {output_dir}")
    print(f"Параметры робота: L={robot.L}, W={robot.W}, R={robot.R}")
    print(f"Масса: {robot.m} кг, Момент инерции: {robot.I} кг⋅м²")
    print(f"Тип робота: (1,2) - передние и задние колеса управляемые")
    print(f"Начальное состояние: [0, 3, 2π/3]")
    print(f"Параметры траектории: R1=7м, δ=2π, α=π/6, t=6с, R2=12м")

if __name__ == "__main__":
    main()
