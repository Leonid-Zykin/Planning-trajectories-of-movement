import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class CoordinatedControl2D:
    """
    Алгоритм стабилизации траекторий для 2D движения методом согласованного управления
    """
    
    def __init__(self, m=1.0, k=0.5, c=0.1):
        """
        Параметры системы:
        m - масса материальной точки
        k - коэффициент жесткости
        c - коэффициент демпфирования
        """
        self.m = m
        self.k = k
        self.c = c
        
    def trajectory_phi1(self, x, y):
        """Первая траектория: окружность"""
        return x**2 + y**2 - 4  # x² + y² = 4
    
    def trajectory_phi2(self, x, y):
        """Вторая траектория: эллипс"""
        return (x-2)**2/9 + (y-1)**2/4 - 1  # (x-2)²/9 + (y-1)²/4 = 1
    
    def trajectory_phi3(self, x, y):
        """Третья траектория: парабола"""
        return y - 0.5*x**2 + 2  # y = 0.5x² - 2
    
    def grad_phi1(self, x, y):
        """Градиент первой траектории"""
        return np.array([2*x, 2*y])
    
    def grad_phi2(self, x, y):
        """Градиент второй траектории"""
        return np.array([2*(x-2)/9, 2*(y-1)/4])
    
    def grad_phi3(self, x, y):
        """Градиент третьей траектории"""
        return np.array([-x, 1])
    
    def coordinated_control_law(self, state, t, s_star, trajectory_phase):
        """
        Закон согласованного управления
        state = [x, y, vx, vy]
        s_star - заданная касательная скорость
        trajectory_phase - номер текущей траектории (1, 2, 3)
        """
        x, y, vx, vy = state
        
        # Выбор траектории в зависимости от фазы
        if trajectory_phase == 1:
            phi = self.trajectory_phi1(x, y)
            grad_phi = self.grad_phi1(x, y)
        elif trajectory_phase == 2:
            phi = self.trajectory_phi2(x, y)
            grad_phi = self.grad_phi2(x, y)
        else:  # trajectory_phase == 3
            phi = self.trajectory_phi3(x, y)
            grad_phi = self.grad_phi3(x, y)
        
        # Нормализация градиента
        grad_norm = np.linalg.norm(grad_phi)
        if grad_norm > 1e-6:
            n = grad_phi / grad_norm
        else:
            n = np.array([1.0, 0.0])
        
        # Касательный вектор (перпендикуляр к нормали)
        tau = np.array([-n[1], n[0]])
        
        # Скорость вдоль касательной
        v_tau = np.dot([vx, vy], tau)
        
        # Ошибка по нормали
        error_normal = phi
        
        # Закон согласованного управления
        # u = -k*phi*n - c*v_normal + s_star*tau
        v_normal = np.dot([vx, vy], n)
        u_normal = -self.k * error_normal * n - self.c * v_normal * n
        u_tangential = s_star * tau
        
        # Общее управление
        u = u_normal + u_tangential
        
        return u
    
    def dynamics(self, state, t, s_star, trajectory_phase):
        """
        Динамика системы с управлением
        """
        x, y, vx, vy = state
        
        # Вычисление управления
        u = self.coordinated_control_law(state, t, s_star, trajectory_phase)
        
        # Динамические уравнения
        dx_dt = vx
        dy_dt = vy
        dvx_dt = u[0] / self.m
        dvy_dt = u[1] / self.m
        
        return [dx_dt, dy_dt, dvx_dt, dvy_dt]
    
    def simulate_trajectory(self, x0, t_span, s_star, trajectory_phases, phase_times):
        """
        Симуляция движения по траекториям
        trajectory_phases - список фаз траекторий [1, 2, 3]
        phase_times - времена переключения между фазами
        """
        def rhs(state, t):
            # Определение текущей фазы траектории
            current_phase = 1
            for i, phase_time in enumerate(phase_times):
                if t >= phase_time:
                    current_phase = trajectory_phases[i+1] if i+1 < len(trajectory_phases) else trajectory_phases[-1]
            
            return self.dynamics(state, t, s_star, current_phase)
        
        sol = odeint(rhs, x0, t_span)
        return sol

def plot_trajectories_and_control(controller, t, sol, s_star, title="Стабилизация траекторий 2D"):
    """Построение графиков траекторий и управления"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    x = sol[:, 0]
    y = sol[:, 1]
    vx = sol[:, 2]
    vy = sol[:, 3]
    
    # Траектория в плоскости
    ax1.plot(x, y, 'b-', linewidth=2, label='Фактическая траектория')
    
    # Построение эталонных траекторий
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 2*np.cos(theta)
    y_circle = 2*np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r--', linewidth=1, label='φ₁: окружность')
    
    theta_ellipse = np.linspace(0, 2*np.pi, 100)
    x_ellipse = 2 + 3*np.cos(theta_ellipse)
    y_ellipse = 1 + 2*np.sin(theta_ellipse)
    ax1.plot(x_ellipse, y_ellipse, 'g--', linewidth=1, label='φ₂: эллипс')
    
    x_parabola = np.linspace(-3, 3, 100)
    y_parabola = 0.5*x_parabola**2 - 2
    ax1.plot(x_parabola, y_parabola, 'm--', linewidth=1, label='φ₃: парабола')
    
    ax1.plot(x[0], y[0], 'go', markersize=8, label='Начало')
    ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='Конец')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Траектория движения (s* = {s_star})')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Скорости
    ax2.plot(t, vx, 'b-', linewidth=2, label='Vx')
    ax2.plot(t, vy, 'r-', linewidth=2, label='Vy')
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Скорость')
    ax2.set_title('Скорости')
    ax2.legend()
    ax2.grid(True)
    
    # Ошибки стабилизации
    phi1_errors = [controller.trajectory_phi1(x[i], y[i]) for i in range(len(t))]
    phi2_errors = [controller.trajectory_phi2(x[i], y[i]) for i in range(len(t))]
    phi3_errors = [controller.trajectory_phi3(x[i], y[i]) for i in range(len(t))]
    
    ax3.plot(t, phi1_errors, 'r-', linewidth=2, label='φ₁ ошибка')
    ax3.plot(t, phi2_errors, 'g-', linewidth=2, label='φ₂ ошибка')
    ax3.plot(t, phi3_errors, 'm-', linewidth=2, label='φ₃ ошибка')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Ошибка стабилизации')
    ax3.set_title('Ошибки стабилизации траекторий')
    ax3.legend()
    ax3.grid(True)
    
    # Управляющие воздействия
    ux = np.gradient(vx, t)
    uy = np.gradient(vy, t)
    ax4.plot(t, ux, 'b-', linewidth=2, label='Ux')
    ax4.plot(t, uy, 'r-', linewidth=2, label='Uy')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('Управление')
    ax4.set_title('Управляющие воздействия')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # Параметры системы
    controller = CoordinatedControl2D(m=1.0, k=0.5, c=0.1)
    
    # Начальные условия
    x0 = np.array([2.5, 0.0, 0.0, 0.0])  # [x, y, vx, vy]
    
    # Временной интервал
    t = np.linspace(0, 20, 1000)
    
    # Фазы траекторий и времена переключения
    trajectory_phases = [1, 2, 3]
    phase_times = [6.67, 13.33]  # Переключения на 1/3 и 2/3 времени
    
    # Скорости для симуляции
    speeds = [1.0, 3.0, 5.0]
    
    for s_star in speeds:
        print(f"Симуляция для s* = {s_star}")
        
        # Симуляция
        sol = controller.simulate_trajectory(x0, t, s_star, trajectory_phases, phase_times)
        
        # Построение графиков
        fig = plot_trajectories_and_control(controller, t, sol, s_star)
        
        # Сохранение результатов
        output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab3/images/task1"
        os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(os.path.join(output_dir, f"coordinated_control_2d_s{s_star}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохранение данных
        np.savez(os.path.join(output_dir, f"coordinated_control_2d_s{s_star}.npz"),
                t=t, x=sol[:, 0], y=sol[:, 1], vx=sol[:, 2], vy=sol[:, 3])
        
        print(f"Результаты для s* = {s_star} сохранены в {output_dir}")

if __name__ == "__main__":
    main()
