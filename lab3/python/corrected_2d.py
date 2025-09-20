import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class CorrectedCoordinatedControl2D:
    """
    Исправленный алгоритм стабилизации траекторий для 2D движения
    """
    
    def __init__(self, m=1.0, k=5.0, c=2.0):
        self.m = m
        self.k = k
        self.c = c
        
    def trajectory_phi1(self, x, y):
        """Окружность: x² + y² = 4"""
        return x**2 + y**2 - 4
    
    def trajectory_phi2(self, x, y):
        """Эллипс: (x-2)²/9 + (y-1)²/4 = 1"""
        return (x-2)**2/9 + (y-1)**2/4 - 1
    
    def trajectory_phi3(self, x, y):
        """Парабола: y = 0.5x² - 2"""
        return y - 0.5*x**2 + 2
    
    def grad_phi1(self, x, y):
        return np.array([2*x, 2*y])
    
    def grad_phi2(self, x, y):
        return np.array([2*(x-2)/9, 2*(y-1)/4])
    
    def grad_phi3(self, x, y):
        return np.array([-x, 1])
    
    def corrected_control_law(self, state, t, s_star, trajectory_phase):
        """
        Исправленный закон согласованного управления
        """
        x, y, vx, vy = state
        
        # Выбор траектории
        if trajectory_phase == 1:
            phi = self.trajectory_phi1(x, y)
            grad_phi = self.grad_phi1(x, y)
        elif trajectory_phase == 2:
            phi = self.trajectory_phi2(x, y)
            grad_phi = self.grad_phi2(x, y)
        else:
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
        
        # Ограничение ошибки для стабильности
        phi_limited = np.clip(phi, -10, 10)
        
        # Скорость вдоль касательной
        v_tau = np.dot([vx, vy], tau)
        
        # Скорость по нормали
        v_normal = np.dot([vx, vy], n)
        
        # Закон согласованного управления
        # u = -k*phi*n - c*v_normal + s_star*tau
        u_normal = -self.k * phi_limited * n - self.c * v_normal * n
        u_tangential = s_star * tau
        
        # Ограничение управления
        u = u_normal + u_tangential
        u = np.clip(u, -20, 20)
        
        return u
    
    def dynamics(self, state, t, s_star, trajectory_phase):
        x, y, vx, vy = state
        u = self.corrected_control_law(state, t, s_star, trajectory_phase)
        
        dx_dt = vx
        dy_dt = vy
        dvx_dt = u[0] / self.m
        dvy_dt = u[1] / self.m
        
        return [dx_dt, dy_dt, dvx_dt, dvy_dt]
    
    def simulate_trajectory(self, x0, t_span, s_star, trajectory_phases, phase_times):
        def rhs(state, t):
            current_phase = 1
            for i, phase_time in enumerate(phase_times):
                if t >= phase_time:
                    current_phase = trajectory_phases[i+1] if i+1 < len(trajectory_phases) else trajectory_phases[-1]
            return self.dynamics(state, t, s_star, current_phase)
        
        sol = odeint(rhs, x0, t_span, rtol=1e-6, atol=1e-8)
        return sol

def plot_corrected_trajectories(controller, t, sol, s_star, title="Исправленная стабилизация 2D"):
    """Построение графиков исправленных траекторий"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    x = sol[:, 0]
    y = sol[:, 1]
    vx = sol[:, 2]
    vy = sol[:, 3]
    
    # Траектория в плоскости
    ax1.plot(x, y, 'b-', linewidth=2, label='Фактическая траектория')
    
    # Эталонные траектории
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = 2*np.cos(theta)
    y_circle = 2*np.sin(theta)
    ax1.plot(x_circle, y_circle, 'r--', linewidth=2, label='φ₁: окружность')
    
    theta_ellipse = np.linspace(0, 2*np.pi, 100)
    x_ellipse = 2 + 3*np.cos(theta_ellipse)
    y_ellipse = 1 + 2*np.sin(theta_ellipse)
    ax1.plot(x_ellipse, y_ellipse, 'g--', linewidth=2, label='φ₂: эллипс')
    
    x_parabola = np.linspace(-3, 3, 100)
    y_parabola = 0.5*x_parabola**2 - 2
    ax1.plot(x_parabola, y_parabola, 'm--', linewidth=2, label='φ₃: парабола')
    
    ax1.plot(x[0], y[0], 'go', markersize=8, label='Начало')
    ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='Конец')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Исправленная траектория (s* = {s_star})')
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
    ax3.set_title('Ошибки стабилизации')
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
    # Исправленные параметры
    controller = CorrectedCoordinatedControl2D(m=1.0, k=5.0, c=2.0)
    
    # Начальные условия ближе к первой траектории
    x0 = np.array([2.1, 0.0, 0.0, 0.0])
    t = np.linspace(0, 20, 2000)  # Увеличиваем количество точек
    
    # Фазы траекторий
    trajectory_phases = [1, 2, 3]
    phase_times = [6.67, 13.33]
    
    speeds = [1.0, 3.0, 5.0]
    
    for s_star in speeds:
        print(f"Симуляция исправленного 2D алгоритма для s* = {s_star}")
        
        # Симуляция
        sol = controller.simulate_trajectory(x0, t, s_star, trajectory_phases, phase_times)
        
        # Проверка ошибок
        x, y = sol[:, 0], sol[:, 1]
        phi1_errors = [controller.trajectory_phi1(x[i], y[i]) for i in range(len(t))]
        phi2_errors = [controller.trajectory_phi2(x[i], y[i]) for i in range(len(t))]
        phi3_errors = [controller.trajectory_phi3(x[i], y[i]) for i in range(len(t))]
        
        print(f"Максимальные ошибки для s* = {s_star}:")
        print(f"  φ₁: {max(abs(e) for e in phi1_errors):.4f}")
        print(f"  φ₂: {max(abs(e) for e in phi2_errors):.4f}")
        print(f"  φ₃: {max(abs(e) for e in phi3_errors):.4f}")
        
        # Построение графиков
        fig = plot_corrected_trajectories(controller, t, sol, s_star)
        
        # Сохранение
        output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab3/images/task1"
        os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(os.path.join(output_dir, f"corrected_2d_s{s_star}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Результаты сохранены для s* = {s_star}")

if __name__ == "__main__":
    main()
