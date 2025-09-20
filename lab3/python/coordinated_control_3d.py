import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class CoordinatedControl3D:
    """
    Алгоритм стабилизации траекторий для 3D движения методом согласованного управления
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
        
    def trajectory_phi1(self, x, y, z):
        """Первая пространственная кривая: сфера"""
        return x**2 + y**2 + z**2 - 4  # x² + y² + z² = 4
    
    def trajectory_phi2(self, x, y, z):
        """Вторая пространственная кривая: цилиндр"""
        return x**2 + y**2 - 1  # x² + y² = 1
    
    def grad_phi1(self, x, y, z):
        """Градиент первой кривой"""
        return np.array([2*x, 2*y, 2*z])
    
    def grad_phi2(self, x, y, z):
        """Градиент второй кривой"""
        return np.array([2*x, 2*y, 0])
    
    def coordinated_control_law(self, state, s_star):
        """
        Закон согласованного управления для 3D
        state = [x, y, z, vx, vy, vz]
        s_star - заданная касательная скорость
        """
        x, y, z, vx, vy, vz = state
        
        # Вычисление градиентов
        grad_phi1 = self.grad_phi1(x, y, z)
        grad_phi2 = self.grad_phi2(x, y, z)
        
        # Нормализация градиентов
        grad1_norm = np.linalg.norm(grad_phi1)
        grad2_norm = np.linalg.norm(grad_phi2)
        
        if grad1_norm > 1e-6:
            n1 = grad_phi1 / grad1_norm
        else:
            n1 = np.array([1.0, 0.0, 0.0])
            
        if grad2_norm > 1e-6:
            n2 = grad_phi2 / grad2_norm
        else:
            n2 = np.array([0.0, 1.0, 0.0])
        
        # Касательный вектор к пересечению кривых
        # Направлен вдоль векторного произведения градиентов
        tau = np.cross(n1, n2)
        tau_norm = np.linalg.norm(tau)
        
        if tau_norm > 1e-6:
            tau = tau / tau_norm
        else:
            # Если градиенты параллельны, выбираем произвольное направление
            tau = np.array([0.0, 0.0, 1.0])
        
        # Скорость вдоль касательной
        v_tau = np.dot([vx, vy, vz], tau)
        
        # Ошибки стабилизации
        error1 = self.trajectory_phi1(x, y, z)
        error2 = self.trajectory_phi2(x, y, z)
        
        # Скорости по нормалям
        v_normal1 = np.dot([vx, vy, vz], n1)
        v_normal2 = np.dot([vx, vy, vz], n2)
        
        # Закон согласованного управления
        u_normal1 = -self.k * error1 * n1 - self.c * v_normal1 * n1
        u_normal2 = -self.k * error2 * n2 - self.c * v_normal2 * n2
        u_tangential = s_star * tau
        
        # Общее управление
        u = u_normal1 + u_normal2 + u_tangential
        
        return u
    
    def dynamics(self, state, t, s_star):
        """
        Динамика системы с управлением
        """
        x, y, z, vx, vy, vz = state
        
        # Вычисление управления
        u = self.coordinated_control_law(state, s_star)
        
        # Динамические уравнения
        dx_dt = vx
        dy_dt = vy
        dz_dt = vz
        dvx_dt = u[0] / self.m
        dvy_dt = u[1] / self.m
        dvz_dt = u[2] / self.m
        
        return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]
    
    def simulate_trajectory(self, x0, t_span, s_star):
        """
        Симуляция движения по 3D траектории
        """
        def rhs(state, t):
            return self.dynamics(state, t, s_star)
        
        sol = odeint(rhs, x0, t_span)
        return sol

def plot_3d_trajectory_and_control(controller, t, sol, s_star, title="Стабилизация траекторий 3D"):
    """Построение графиков 3D траекторий и управления"""
    fig = plt.figure(figsize=(20, 15))
    
    x = sol[:, 0]
    y = sol[:, 1]
    z = sol[:, 2]
    vx = sol[:, 3]
    vy = sol[:, 4]
    vz = sol[:, 5]
    
    # 3D траектория
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=2, label='Фактическая траектория')
    
    # Построение эталонных поверхностей
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    X_sphere = 2 * np.outer(np.cos(u), np.sin(v))
    Y_sphere = 2 * np.outer(np.sin(u), np.sin(v))
    Z_sphere = 2 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.3, color='red')
    
    # Цилиндр
    theta = np.linspace(0, 2*np.pi, 20)
    z_cyl = np.linspace(-3, 3, 20)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    X_cyl = np.cos(theta_grid)
    Y_cyl = np.sin(theta_grid)
    Z_cyl = z_grid
    ax1.plot_surface(X_cyl, Y_cyl, Z_cyl, alpha=0.3, color='green')
    
    ax1.plot(x[0], y[0], z[0], 'go', markersize=8, label='Начало')
    ax1.plot(x[-1], y[-1], z[-1], 'ro', markersize=8, label='Конец')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'3D траектория (s* = {s_star})')
    ax1.legend()
    
    # Проекции на плоскости
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x, y, 'b-', linewidth=2, label='XY проекция')
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'g--', linewidth=1, label='Цилиндр')
    ax2.plot(2*np.cos(theta_circle), 2*np.sin(theta_circle), 'r--', linewidth=1, label='Сфера')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Проекция XY')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x, z, 'b-', linewidth=2, label='XZ проекция')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Проекция XZ')
    ax3.legend()
    ax3.grid(True)
    
    # Скорости
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(t, vx, 'b-', linewidth=2, label='Vx')
    ax4.plot(t, vy, 'r-', linewidth=2, label='Vy')
    ax4.plot(t, vz, 'g-', linewidth=2, label='Vz')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('Скорость')
    ax4.set_title('Скорости')
    ax4.legend()
    ax4.grid(True)
    
    # Ошибки стабилизации
    phi1_errors = [controller.trajectory_phi1(x[i], y[i], z[i]) for i in range(len(t))]
    phi2_errors = [controller.trajectory_phi2(x[i], y[i], z[i]) for i in range(len(t))]
    
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(t, phi1_errors, 'r-', linewidth=2, label='φ₁ ошибка (сфера)')
    ax5.plot(t, phi2_errors, 'g-', linewidth=2, label='φ₂ ошибка (цилиндр)')
    ax5.set_xlabel('Время')
    ax5.set_ylabel('Ошибка стабилизации')
    ax5.set_title('Ошибки стабилизации')
    ax5.legend()
    ax5.grid(True)
    
    # Управляющие воздействия
    ux = np.gradient(vx, t)
    uy = np.gradient(vy, t)
    uz = np.gradient(vz, t)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(t, ux, 'b-', linewidth=2, label='Ux')
    ax6.plot(t, uy, 'r-', linewidth=2, label='Uy')
    ax6.plot(t, uz, 'g-', linewidth=2, label='Uz')
    ax6.set_xlabel('Время')
    ax6.set_ylabel('Управление')
    ax6.set_title('Управляющие воздействия')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    return fig

def main():
    # Параметры системы
    controller = CoordinatedControl3D(m=1.0, k=0.5, c=0.1)
    
    # Начальные условия
    x0 = np.array([1.5, 0.0, 1.0, 0.0, 0.0, 0.0])  # [x, y, z, vx, vy, vz]
    
    # Временной интервал
    t = np.linspace(0, 15, 1000)
    
    # Скорости для симуляции
    speeds = [1.0, 3.0, 5.0]
    
    for s_star in speeds:
        print(f"Симуляция 3D для s* = {s_star}")
        
        # Симуляция
        sol = controller.simulate_trajectory(x0, t, s_star)
        
        # Построение графиков
        fig = plot_3d_trajectory_and_control(controller, t, sol, s_star)
        
        # Сохранение результатов
        output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab3/images/task2"
        os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(os.path.join(output_dir, f"coordinated_control_3d_s{s_star}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Сохранение данных
        np.savez(os.path.join(output_dir, f"coordinated_control_3d_s{s_star}.npz"),
                t=t, x=sol[:, 0], y=sol[:, 1], z=sol[:, 2], 
                vx=sol[:, 3], vy=sol[:, 4], vz=sol[:, 5])
        
        print(f"Результаты 3D для s* = {s_star} сохранены в {output_dir}")

if __name__ == "__main__":
    main()
