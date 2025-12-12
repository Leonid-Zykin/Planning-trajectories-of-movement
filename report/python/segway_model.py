"""
Модель Segway (неполноприводная система - перевернутый маятник на колесах)
Динамическая модель для свободного и управляемого движения
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

class SegwayModel:
    """
    Модель Segway как неполноприводной системы.
    
    Состояние: [theta, theta_dot, phi, phi_dot, x, x_dot]
    где:
    - theta: угол наклона платформы (pitch) относительно вертикали
    - phi: угол поворота колес
    - x: горизонтальное положение
    
    Параметры:
    - M: масса платформы
    - m: масса колес
    - L: расстояние от оси колес до центра масс платформы
    - R: радиус колес
    - I_p: момент инерции платформы относительно оси колес
    - I_w: момент инерции колес
    - g: ускорение свободного падения
    """
    
    def __init__(self, M=10.0, m=2.0, L=0.5, R=0.1, I_p=1.0, I_w=0.1, g=9.81):
        self.M = M  # Масса платформы (кг)
        self.m = m  # Масса колес (кг)
        self.L = L  # Расстояние от оси до центра масс (м)
        self.R = R  # Радиус колес (м)
        self.I_p = I_p  # Момент инерции платформы (кг·м²)
        self.I_w = I_w  # Момент инерции колес (кг·м²)
        self.g = g  # Ускорение свободного падения (м/с²)
        
    def dynamics_free(self, state, t):
        """
        Динамика свободного движения (без управления).
        
        Args:
            state: [theta, theta_dot, phi, phi_dot, x, x_dot]
            t: время
            
        Returns:
            производные состояния
        """
        theta, theta_dot, phi, phi_dot, x, x_dot = state
        
        # Упрощенная модель без управления
        # Уравнения Лагранжа для перевернутого маятника
        
        # Матрица масс
        M11 = self.I_p + self.M * self.L**2
        M12 = self.M * self.L * self.R * np.cos(theta)
        M21 = M12
        M22 = self.I_w + (self.M + self.m) * self.R**2
        
        # Вектор Кориолиса и центробежных сил
        C1 = -self.M * self.L * self.R * np.sin(theta) * phi_dot**2
        C2 = -self.M * self.L * self.R * np.sin(theta) * theta_dot * phi_dot
        
        # Вектор гравитационных сил
        G1 = -self.M * self.g * self.L * np.sin(theta)
        G2 = 0.0
        
        # Без управления u = 0
        u = 0.0
        
        # Решение системы уравнений
        M_mat = np.array([[M11, M12], [M21, M22]])
        rhs = np.array([C1 + G1 + u, C2 + G2 - u])
        
        accel = np.linalg.solve(M_mat, rhs)
        theta_ddot = accel[0]
        phi_ddot = accel[1]
        
        # Кинематические связи
        x_dot = self.R * phi_dot
        x_ddot = self.R * phi_ddot
        
        return [theta_dot, theta_ddot, phi_dot, phi_ddot, x_dot, x_ddot]
    
    def dynamics_controlled(self, state, t, controller_func):
        """
        Динамика управляемого движения.
        
        Args:
            state: [theta, theta_dot, phi, phi_dot, x, x_dot]
            t: время
            controller_func: функция контроллера controller_func(state, t) -> u
            
        Returns:
            производные состояния
        """
        theta, theta_dot, phi, phi_dot, x, x_dot = state
        
        # Вычисление управляющего воздействия
        u = controller_func(state, t)
        
        # Матрица масс
        M11 = self.I_p + self.M * self.L**2
        M12 = self.M * self.L * self.R * np.cos(theta)
        M21 = M12
        M22 = self.I_w + (self.M + self.m) * self.R**2
        
        # Вектор Кориолиса и центробежных сил
        C1 = -self.M * self.L * self.R * np.sin(theta) * phi_dot**2
        C2 = -self.M * self.L * self.R * np.sin(theta) * theta_dot * phi_dot
        
        # Вектор гравитационных сил
        G1 = -self.M * self.g * self.L * np.sin(theta)
        G2 = 0.0
        
        # Решение системы уравнений
        M_mat = np.array([[M11, M12], [M21, M22]])
        rhs = np.array([C1 + G1 + u, C2 + G2 - u])
        
        accel = np.linalg.solve(M_mat, rhs)
        theta_ddot = accel[0]
        phi_ddot = accel[1]
        
        # Кинематические связи
        x_dot = self.R * phi_dot
        x_ddot = self.R * phi_ddot
        
        return [theta_dot, theta_ddot, phi_dot, phi_ddot, x_dot, x_ddot]
    
    def simulate_free(self, initial_state, t_span):
        """
        Симуляция свободного движения.
        
        Args:
            initial_state: начальное состояние [theta, theta_dot, phi, phi_dot, x, x_dot]
            t_span: массив времени
            
        Returns:
            решение ODE
        """
        sol = odeint(self.dynamics_free, initial_state, t_span)
        return sol
    
    def simulate_controlled(self, initial_state, t_span, controller_func):
        """
        Симуляция управляемого движения.
        
        Args:
            initial_state: начальное состояние
            t_span: массив времени
            controller_func: функция контроллера
            
        Returns:
            решение ODE
        """
        def dynamics_wrapper(state, t):
            return self.dynamics_controlled(state, t, controller_func)
        
        sol = odeint(dynamics_wrapper, initial_state, t_span)
        return sol

