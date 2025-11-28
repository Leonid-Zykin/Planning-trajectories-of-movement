import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class CorrectedCoordinatedControl3D:
    """
    Алгоритм стабилизации 3D траекторий с проекцией на пересечение сферы и цилиндра.
    """
    
    def __init__(self, m=1.0, k=300.0, c=80.0, k_tau=8.0, d_tau=3.0,
                 adaptive_gain=0.6, max_tangential_speed=2.5, u_limit=120.0,
                 plane_sign=1.0):
        self.m = m
        self.k = k
        self.c = c
        self.k_tau = k_tau
        self.d_tau = d_tau
        self.adaptive_gain = adaptive_gain
        self.max_tangential_speed = max_tangential_speed
        self.u_limit = u_limit
        self.set_plane_sign(plane_sign)
        
    def trajectory_phi1(self, x, y, z):
        """Сфера: x² + y² + z² = 4"""
        return x**2 + y**2 + z**2 - 4
    
    def trajectory_phi2(self, x, y, z):
        """Цилиндр: x² + y² = 1"""
        return x**2 + y**2 - 1
    
    def grad_phi1(self, x, y, z):
        return np.array([2*x, 2*y, 2*z])
    
    def grad_phi2(self, x, y, z):
        return np.array([2*x, 2*y, 0])
    
    def set_plane_sign(self, sign):
        self.plane_sign = 1.0 if sign >= 0 else -1.0
        self.z_plane = self.plane_sign * np.sqrt(3)
    
    def corrected_control_law(self, state, s_star):
        """
        Закон управления с проекцией на пересечение сферы и цилиндра.
        """
        x, y, z, vx, vy, vz = state
        theta_ref = np.arctan2(y, x)
        pos_ref = np.array([np.cos(theta_ref), np.sin(theta_ref), self.z_plane])
        pos = np.array([x, y, z])
        vel = np.array([vx, vy, vz])
        error_vec = pos - pos_ref
        
        radial_dir = np.array([np.cos(theta_ref), np.sin(theta_ref), 0.0])
        vertical_dir = np.array([0.0, 0.0, self.plane_sign])
        
        e_radial = np.dot(error_vec, radial_dir)
        e_vertical = np.dot(error_vec, vertical_dir)
        v_radial = np.dot(vel, radial_dir)
        v_vertical = np.dot(vel, vertical_dir)
        
        u_radial = -self.k * e_radial * radial_dir - self.c * v_radial * radial_dir
        u_vertical = -self.k * e_vertical * vertical_dir - self.c * v_vertical * vertical_dir
        
        tau = np.array([-np.sin(theta_ref), np.cos(theta_ref), 0.0])
        v_tau = np.dot(vel, tau)
        dominant_error = max(abs(e_radial), abs(e_vertical))
        s_effective = s_star * np.exp(-self.adaptive_gain * dominant_error)
        s_effective = np.clip(s_effective, -self.max_tangential_speed, self.max_tangential_speed)
        tangential_error = s_effective - v_tau
        u_tangential = (self.k_tau * tangential_error - self.d_tau * v_tau) * tau
        
        u = u_radial + u_vertical + u_tangential
        u = np.clip(u, -self.u_limit, self.u_limit)
        return u
    
    def dynamics(self, state, t, s_star):
        x, y, z, vx, vy, vz = state
        u = self.corrected_control_law(state, s_star)
        
        dx_dt = vx
        dy_dt = vy
        dz_dt = vz
        dvx_dt = u[0] / self.m
        dvy_dt = u[1] / self.m
        dvz_dt = u[2] / self.m
        
        return [dx_dt, dy_dt, dz_dt, dvx_dt, dvy_dt, dvz_dt]
    
    def simulate_trajectory(self, x0, t_span, s_star):
        def rhs(state, t):
            return self.dynamics(state, t, s_star)
        
        sol = odeint(rhs, x0, t_span, rtol=1e-6, atol=1e-8)
        return sol

def plot_corrected_3d_trajectory(controller, t, sol, s_star, title="Исправленная стабилизация 3D"):
    """Построение графиков исправленных 3D траекторий"""
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
    
    # Эталонные поверхности
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
    ax1.set_title(f'Исправленная 3D траектория (s* = {s_star})')
    ax1.legend()
    
    # Проекции
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
    controller = CorrectedCoordinatedControl3D()
    
    # Начальные условия на пересечении поверхностей
    x0 = np.array([1.0, 0.0, np.sqrt(3), 0.0, 0.0, 0.0])
    controller.set_plane_sign(np.sign(x0[2]))
    t = np.linspace(0, 15, 1500)
    
    speeds = [1.0, 3.0, 5.0]
    
    for s_star in speeds:
        print(f"Симуляция исправленного 3D алгоритма для s* = {s_star}")
        
        # Симуляция
        sol = controller.simulate_trajectory(x0, t, s_star)
        
        # Проверка ошибок
        x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]
        phi1_errors = [controller.trajectory_phi1(x[i], y[i], z[i]) for i in range(len(t))]
        phi2_errors = [controller.trajectory_phi2(x[i], y[i], z[i]) for i in range(len(t))]
        
        print(f"Максимальные ошибки для s* = {s_star}:")
        print(f"  φ₁ (сфера): {max(abs(e) for e in phi1_errors):.4f}")
        print(f"  φ₂ (цилиндр): {max(abs(e) for e in phi2_errors):.4f}")
        
        # Построение графиков
        fig = plot_corrected_3d_trajectory(controller, t, sol, s_star)
        
        # Сохранение
        output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab3/images/task2"
        os.makedirs(output_dir, exist_ok=True)
        
        fig.savefig(os.path.join(output_dir, f"corrected_3d_s{s_star}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Результаты сохранены для s* = {s_star}")

if __name__ == "__main__":
    main()
