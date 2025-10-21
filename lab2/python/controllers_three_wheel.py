#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контроллеры для трехколесного мобильного робота типа (1,2)
Статическая и динамическая линеаризация обратной связи
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ThreeWheelController:
    """Контроллер для трехколесного мобильного робота"""
    
    def __init__(self):
        # Геометрические параметры
        self.L = 0.3  # База робота
        self.W = 0.2  # Колея
        self.R = 0.05  # Радиус колес
        self.d = 0.1   # Геометрический параметр
        
        # Параметры контроллеров
        self.kp1, self.kp2 = 2.0, 2.0  # Коэффициенты пропорционального управления
        self.kd1, self.kd2 = 1.5, 1.5  # Коэффициенты дифференциального управления
        self.ki1, self.ki2 = 0.5, 0.5  # Коэффициенты интегрального управления
        
        # Интегральные ошибки
        self.integral_error = np.zeros(2)
        
    def rotation_matrix(self, theta):
        """Матрица поворота R(θ)"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    
    def sigma_matrix(self, beta_s1, beta_s2):
        """Матрица Σ(βs1, βs2) согласно методичке"""
        return np.array([
            [np.cos(beta_s1), np.cos(beta_s2)],
            [np.sin(beta_s1), np.sin(beta_s2)],
            [np.sin(beta_s2 - beta_s1) / self.L, np.sin(beta_s1 - beta_s2) / self.L]
        ])
    
    def static_linearization_controller(self, state, ref_state, ref_vel):
        """
        Статическая линеаризация обратной связи
        
        Args:
            state: [x, y, theta, beta_s1, beta_s2] - текущее состояние
            ref_state: [x_ref, y_ref, theta_ref] - желаемое состояние
            ref_vel: [vx_ref, vy_ref, omega_ref] - желаемые скорости
        
        Returns:
            eta: [eta1, eta2] - обобщенные скорости
            zeta: [zeta1, zeta2] - скорости изменения углов рулевых колес
        """
        x, y, theta, beta_s1, beta_s2 = state
        x_ref, y_ref, theta_ref = ref_state
        vx_ref, vy_ref, omega_ref = ref_vel
        
        # Ошибки состояния
        e_x = x_ref - x
        e_y = y_ref - y
        e_theta = theta_ref - theta
        
        # Матрица поворота
        R = self.rotation_matrix(theta)
        
        # Матрица Σ(βs1, βs2)
        Sigma = self.sigma_matrix(beta_s1, beta_s2)
        
        # Желаемые скорости в локальной системе координат
        xi_ref_dot = np.array([vx_ref, vy_ref, omega_ref])
        
        # Статическая линеаризация: η = (R^T Σ)^+ (ξ̇_ref + K_p e)
        try:
            # Псевдообратная матрица
            R_Sigma = R.T @ Sigma
            R_Sigma_pinv = np.linalg.pinv(R_Sigma)
            
            # Закон управления
            eta = R_Sigma_pinv @ (xi_ref_dot + np.array([self.kp1 * e_x, self.kp2 * e_y, 0.5 * e_theta]))
            
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем простой закон
            eta = np.array([vx_ref, vy_ref])
        
        # Простой закон для углов рулевых колес
        zeta = np.array([self.kp1 * e_theta, self.kp2 * e_theta])
        
        return eta, zeta
    
    def dynamics(self, state, t, eta, zeta):
        """
        Динамика трехколесного робота согласно уравнениям (20) и (21)
        
        Args:
            state: [x, y, theta, beta_s1, beta_s2] - состояние робота
            t: время
            eta: [eta1, eta2] - обобщенные скорости
            zeta: [zeta1, zeta2] - скорости изменения углов рулевых колес
        
        Returns:
            dstate: производная состояния
        """
        x, y, theta, beta_s1, beta_s2 = state
        
        # Матрица поворота
        R = self.rotation_matrix(theta)
        
        # Матрица Σ(βs1, βs2)
        Sigma = self.sigma_matrix(beta_s1, beta_s2)
        
        # Уравнение (20): ξ̇ = R^T(θ)Σ(βs1, βs2)η
        xi_dot = R.T @ Sigma @ eta
        
        # Уравнение (21): β̇s = ζ
        beta_dot = zeta
        
        return np.concatenate([xi_dot, beta_dot])
    
    def dynamic_linearization_controller(self, state, ref_state, ref_vel, ref_acc, dt):
        """
        Динамическая линеаризация обратной связи с ПИД-управлением
        
        Args:
            state: [x, y, theta, beta_s1, beta_s2] - текущее состояние
            ref_state: [x_ref, y_ref, theta_ref] - желаемое состояние
            ref_vel: [vx_ref, vy_ref, omega_ref] - желаемые скорости
            ref_acc: [ax_ref, ay_ref, alpha_ref] - желаемые ускорения
            dt: шаг времени
        
        Returns:
            eta: [eta1, eta2] - обобщенные скорости
            zeta: [zeta1, zeta2] - скорости изменения углов рулевых колес
        """
        x, y, theta, beta_s1, beta_s2 = state
        x_ref, y_ref, theta_ref = ref_state
        vx_ref, vy_ref, omega_ref = ref_vel
        ax_ref, ay_ref, alpha_ref = ref_acc
        
        # Ошибки состояния
        e_x = x_ref - x
        e_y = y_ref - y
        e_theta = theta_ref - theta
        
        # Обновление интегральных ошибок
        self.integral_error[0] += e_x * dt
        self.integral_error[1] += e_y * dt
        
        # Матрица поворота
        R = self.rotation_matrix(theta)
        
        # Матрица Σ(βs1, βs2)
        Sigma = self.sigma_matrix(beta_s1, beta_s2)
        
        # Желаемые ускорения в локальной системе координат
        xi_ref_ddot = np.array([ax_ref, ay_ref, alpha_ref])
        
        # Динамическая линеаризация: η = (R^T Σ)^+ (ξ̈_ref + K_p e + K_d ė + K_i ∫e)
        try:
            # Псевдообратная матрица
            R_Sigma = R.T @ Sigma
            R_Sigma_pinv = np.linalg.pinv(R_Sigma)
            
            # ПИД-управление
            pid_term = np.array([
                self.kp1 * e_x + self.ki1 * self.integral_error[0],
                self.kp2 * e_y + self.ki2 * self.integral_error[1],
                0.5 * e_theta
            ])
            
            # Закон управления
            eta = R_Sigma_pinv @ (xi_ref_ddot + pid_term)
            
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем простой закон
            eta = np.array([ax_ref, ay_ref])
        
        # ПИД-управление для углов рулевых колес
        zeta = np.array([
            self.kp1 * e_theta + self.ki1 * self.integral_error[0],
            self.kp2 * e_theta + self.ki2 * self.integral_error[1]
        ])
        
        return eta, zeta

def simulate_robot_tracking_three_wheel():
    """Симуляция слежения за траекторией трехколесным роботом"""
    
    # Создание робота и контроллера
    robot = ThreeWheelController()
    
    # Генерация опорной траектории
    t_ref = np.linspace(0, 50, 500)
    
    # Простая траектория для демонстрации
    x_ref = 2 * np.sin(0.2 * t_ref)
    y_ref = 2 * np.cos(0.2 * t_ref)
    theta_ref = 0.2 * t_ref
    
    vx_ref = 0.4 * np.cos(0.2 * t_ref)
    vy_ref = -0.4 * np.sin(0.2 * t_ref)
    omega_ref = 0.2 * np.ones_like(t_ref)
    
    ax_ref = -0.08 * np.sin(0.2 * t_ref)
    ay_ref = -0.08 * np.cos(0.2 * t_ref)
    alpha_ref = np.zeros_like(t_ref)
    
    # Интерполяция опорных сигналов
    x_ref_func = interp1d(t_ref, x_ref, kind='linear', fill_value='extrapolate')
    y_ref_func = interp1d(t_ref, y_ref, kind='linear', fill_value='extrapolate')
    theta_ref_func = interp1d(t_ref, theta_ref, kind='linear', fill_value='extrapolate')
    vx_ref_func = interp1d(t_ref, vx_ref, kind='linear', fill_value='extrapolate')
    vy_ref_func = interp1d(t_ref, vy_ref, kind='linear', fill_value='extrapolate')
    omega_ref_func = interp1d(t_ref, omega_ref, kind='linear', fill_value='extrapolate')
    ax_ref_func = interp1d(t_ref, ax_ref, kind='linear', fill_value='extrapolate')
    ay_ref_func = interp1d(t_ref, ay_ref, kind='linear', fill_value='extrapolate')
    alpha_ref_func = interp1d(t_ref, alpha_ref, kind='linear', fill_value='extrapolate')
    
    # Начальное состояние
    initial_state = np.array([0.0, 2.0, 0.0, 0.0, 0.0])  # [x, y, theta, beta_s1, beta_s2]
    
    # Параметры симуляции
    dt = 0.1
    t_sim = np.arange(0, 50, dt)
    
    # Симуляция со статической линеаризацией
    states_static = np.zeros((len(t_sim), 5))
    states_static[0] = initial_state
    
    for i in range(1, len(t_sim)):
        current_state = states_static[i-1]
        
        # Опорные значения
        ref_state = np.array([x_ref_func(t_sim[i]), y_ref_func(t_sim[i]), theta_ref_func(t_sim[i])])
        ref_vel = np.array([vx_ref_func(t_sim[i]), vy_ref_func(t_sim[i]), omega_ref_func(t_sim[i])])
        
        # Управление
        eta, zeta = robot.static_linearization_controller(current_state, ref_state, ref_vel)
        
        # Интеграция
        def ode_func(state, t):
            return robot.dynamics(state, t, eta, zeta)
        
        # Простой метод Эйлера
        state_dot = ode_func(current_state, t_sim[i])
        states_static[i] = current_state + state_dot * dt
    
    # Симуляция с динамической линеаризацией
    robot_dynamic = ThreeWheelController()  # Новый экземпляр для сброса интегральных ошибок
    states_dynamic = np.zeros((len(t_sim), 5))
    states_dynamic[0] = initial_state
    
    for i in range(1, len(t_sim)):
        current_state = states_dynamic[i-1]
        
        # Опорные значения
        ref_state = np.array([x_ref_func(t_sim[i]), y_ref_func(t_sim[i]), theta_ref_func(t_sim[i])])
        ref_vel = np.array([vx_ref_func(t_sim[i]), vy_ref_func(t_sim[i]), omega_ref_func(t_sim[i])])
        ref_acc = np.array([ax_ref_func(t_sim[i]), ay_ref_func(t_sim[i]), alpha_ref_func(t_sim[i])])
        
        # Управление
        eta, zeta = robot_dynamic.dynamic_linearization_controller(current_state, ref_state, ref_vel, ref_acc, dt)
        
        # Интеграция
        def ode_func(state, t):
            return robot_dynamic.dynamics(state, t, eta, zeta)
        
        # Простой метод Эйлера
        state_dot = ode_func(current_state, t_sim[i])
        states_dynamic[i] = current_state + state_dot * dt
    
    return t_sim, states_static, states_dynamic, t_ref, x_ref, y_ref, theta_ref

def plot_comparison_three_wheel():
    """Построение сравнения контроллеров для трехколесного робота"""
    
    # Симуляция
    t_sim, states_static, states_dynamic, t_ref, x_ref, y_ref, theta_ref = simulate_robot_tracking_three_wheel()
    
    # Создание фигуры
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # График траекторий
    ax1 = axes[0, 0]
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, label='Опорная траектория')
    ax1.plot(states_static[:, 0], states_static[:, 1], 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax1.plot(states_dynamic[:, 0], states_dynamic[:, 1], 'r-', linewidth=1.5, label='Динамическая линеаризация')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектории движения')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # График ошибок по X
    ax2 = axes[0, 1]
    x_ref_interp = np.interp(t_sim, t_ref, x_ref)
    error_x_static = x_ref_interp - states_static[:, 0]
    error_x_dynamic = x_ref_interp - states_dynamic[:, 0]
    ax2.plot(t_sim, error_x_static, 'b-', label='Статическая линеаризация')
    ax2.plot(t_sim, error_x_dynamic, 'r-', label='Динамическая линеаризация')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Ошибка по X (м)')
    ax2.set_title('Ошибки слежения по X')
    ax2.legend()
    ax2.grid(True)
    
    # График ошибок по Y
    ax3 = axes[1, 0]
    y_ref_interp = np.interp(t_sim, t_ref, y_ref)
    error_y_static = y_ref_interp - states_static[:, 1]
    error_y_dynamic = y_ref_interp - states_dynamic[:, 1]
    ax3.plot(t_sim, error_y_static, 'b-', label='Статическая линеаризация')
    ax3.plot(t_sim, error_y_dynamic, 'r-', label='Динамическая линеаризация')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Ошибка по Y (м)')
    ax3.set_title('Ошибки слежения по Y')
    ax3.legend()
    ax3.grid(True)
    
    # График углов рулевых колес
    ax4 = axes[1, 1]
    ax4.plot(t_sim, states_static[:, 3], 'b-', label='βs1 (статическая)')
    ax4.plot(t_sim, states_static[:, 4], 'b--', label='βs2 (статическая)')
    ax4.plot(t_sim, states_dynamic[:, 3], 'r-', label='βs1 (динамическая)')
    ax4.plot(t_sim, states_dynamic[:, 4], 'r--', label='βs2 (динамическая)')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Угол (рад)')
    ax4.set_title('Углы рулевых колес')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task2', exist_ok=True)
    plt.savefig('../images/task2/controllers_comparison_three_wheel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Расчет статистики ошибок
    error_norm_static = np.sqrt(error_x_static**2 + error_y_static**2)
    error_norm_dynamic = np.sqrt(error_x_dynamic**2 + error_y_dynamic**2)
    
    print("📊 Статистика ошибок для трехколесного робота:")
    print(f"Статическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_static):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_static):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_static):.4f} м")
    
    print(f"Динамическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_dynamic):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_dynamic):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_dynamic):.4f} м")
    
    print("✅ Сравнение контроллеров для трехколесного робота сохранено: controllers_comparison_three_wheel.png")

if __name__ == "__main__":
    print("🚀 Симуляция контроллеров для трехколесного мобильного робота")
    plot_comparison_three_wheel()
