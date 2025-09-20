import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

class ImprovedCoordinatedControl2D:
    """
    Улучшенный алгоритм стабилизации траекторий для 2D движения
    """
    
    def __init__(self, m=1.0, k=2.0, c=0.5):
        """
        Улучшенные параметры:
        k - увеличенный коэффициент жесткости
        c - увеличенный коэффициент демпфирования
        """
        self.m = m
        self.k = k
        self.c = c
        
    def trajectory_phi1(self, x, y):
        """Первая траектория: окружность"""
        return x**2 + y**2 - 4
    
    def trajectory_phi2(self, x, y):
        """Вторая траектория: эллипс"""
        return (x-2)**2/9 + (y-1)**2/4 - 1
    
    def trajectory_phi3(self, x, y):
        """Третья траектория: парабола"""
        return y - 0.5*x**2 + 2
    
    def grad_phi1(self, x, y):
        return np.array([2*x, 2*y])
    
    def grad_phi2(self, x, y):
        return np.array([2*(x-2)/9, 2*(y-1)/4])
    
    def grad_phi3(self, x, y):
        return np.array([-x, 1])
    
    def improved_control_law(self, state, t, s_star, trajectory_phase):
        """
        Улучшенный закон согласованного управления
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
        
        # Касательный вектор
        tau = np.array([-n[1], n[0]])
        
        # Адаптивные коэффициенты в зависимости от ошибки
        error_magnitude = abs(phi)
        k_adaptive = self.k * (1 + 0.5 * error_magnitude)
        c_adaptive = self.c * (1 + 0.2 * error_magnitude)
        
        # Скорость вдоль касательной
        v_tau = np.dot([vx, vy], tau)
        
        # Ошибка по нормали
        error_normal = phi
        
        # Улучшенный закон управления
        v_normal = np.dot([vx, vy], n)
        u_normal = -k_adaptive * error_normal * n - c_adaptive * v_normal * n
        u_tangential = s_star * tau
        
        # Общее управление
        u = u_normal + u_tangential
        
        return u
    
    def dynamics(self, state, t, s_star, trajectory_phase):
        x, y, vx, vy = state
        u = self.improved_control_law(state, t, s_star, trajectory_phase)
        
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
        
        sol = odeint(rhs, x0, t_span)
        return sol

def test_improved_algorithm():
    """Тест улучшенного алгоритма"""
    controller = ImprovedCoordinatedControl2D(m=1.0, k=2.0, c=0.5)
    
    # Начальные условия ближе к первой траектории
    x0 = np.array([2.1, 0.0, 0.0, 0.0])
    t = np.linspace(0, 20, 1000)
    trajectory_phases = [1, 2, 3]
    phase_times = [6.67, 13.33]
    
    sol = controller.simulate_trajectory(x0, t, 3.0, trajectory_phases, phase_times)
    
    # Проверим ошибки стабилизации
    x, y = sol[:, 0], sol[:, 1]
    phi1_errors = [controller.trajectory_phi1(x[i], y[i]) for i in range(len(t))]
    phi2_errors = [controller.trajectory_phi2(x[i], y[i]) for i in range(len(t))]
    phi3_errors = [controller.trajectory_phi3(x[i], y[i]) for i in range(len(t))]
    
    print('УЛУЧШЕННЫЙ 2D АЛГОРИТМ:')
    print('Максимальные ошибки стабилизации:')
    print(f'φ₁ (окружность): {max(abs(e) for e in phi1_errors):.4f}')
    print(f'φ₂ (эллипс): {max(abs(e) for e in phi2_errors):.4f}')
    print(f'φ₃ (парабола): {max(abs(e) for e in phi3_errors):.4f}')
    
    print(f'Финальные ошибки:')
    print(f'φ₁: {phi1_errors[-1]:.4f}')
    print(f'φ₂: {phi2_errors[-1]:.4f}')
    print(f'φ₃: {phi3_errors[-1]:.4f}')

if __name__ == "__main__":
    test_improved_algorithm()
