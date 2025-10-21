#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенные контроллеры для трехколесного мобильного робота
С правильной траекторией и настройкой параметров
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ImprovedThreeWheelController:
    """Улучшенный контроллер для трехколесного мобильного робота"""
    
    def __init__(self):
        # Геометрические параметры
        self.L = 0.3  # База робота
        self.W = 0.2  # Колея
        self.R = 0.05  # Радиус колес
        self.d = 0.1   # Геометрический параметр
        
        # Улучшенные параметры контроллеров
        self.kp1, self.kp2 = 5.0, 5.0  # Увеличенные коэффициенты пропорционального управления
        self.kd1, self.kd2 = 3.0, 3.0  # Увеличенные коэффициенты дифференциального управления
        self.ki1, self.ki2 = 1.0, 1.0  # Увеличенные коэффициенты интегрального управления
        
        # Интегральные ошибки
        self.integral_error = np.zeros(2)
        self.prev_error = np.zeros(2)
        
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
    
    def dynamics(self, state, t, eta, zeta):
        """Динамика трехколесного робота"""
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
    
    def improved_static_controller(self, state, ref_state, ref_vel):
        """Улучшенная статическая линеаризация"""
        x, y, theta, beta_s1, beta_s2 = state
        x_ref, y_ref, theta_ref = ref_state
        vx_ref, vy_ref, omega_ref = ref_vel
        
        # Ошибки состояния
        e_x = x_ref - x
        e_y = y_ref - y
        e_theta = theta_ref - theta
        
        # Нормализация угловой ошибки
        while e_theta > np.pi:
            e_theta -= 2*np.pi
        while e_theta < -np.pi:
            e_theta += 2*np.pi
        
        # Матрица поворота
        R = self.rotation_matrix(theta)
        
        # Матрица Σ(βs1, βs2)
        Sigma = self.sigma_matrix(beta_s1, beta_s2)
        
        # Желаемые скорости в локальной системе координат
        xi_ref_dot = np.array([vx_ref, vy_ref, omega_ref])
        
        # Улучшенная статическая линеаризация
        try:
            R_Sigma = R.T @ Sigma
            R_Sigma_pinv = np.linalg.pinv(R_Sigma)
            
            # Адаптивные коэффициенты
            kp_adaptive = self.kp1 * (1 + 0.1 * np.sqrt(e_x**2 + e_y**2))
            
            # Закон управления
            eta = R_Sigma_pinv @ (xi_ref_dot + np.array([
                kp_adaptive * e_x, 
                kp_adaptive * e_y, 
                2.0 * e_theta
            ]))
            
        except np.linalg.LinAlgError:
            eta = np.array([vx_ref, vy_ref])
        
        # Улучшенный закон для углов рулевых колес
        zeta = np.array([
            self.kp1 * e_theta + 0.5 * e_x,
            self.kp2 * e_theta - 0.5 * e_y
        ])
        
        return eta, zeta
    
    def improved_dynamic_controller(self, state, ref_state, ref_vel, ref_acc, dt):
        """Улучшенная динамическая линеаризация с ПИД-управлением"""
        x, y, theta, beta_s1, beta_s2 = state
        x_ref, y_ref, theta_ref = ref_state
        vx_ref, vy_ref, omega_ref = ref_vel
        ax_ref, ay_ref, alpha_ref = ref_acc
        
        # Ошибки состояния
        e_x = x_ref - x
        e_y = y_ref - y
        e_theta = theta_ref - theta
        
        # Нормализация угловой ошибки
        while e_theta > np.pi:
            e_theta -= 2*np.pi
        while e_theta < -np.pi:
            e_theta += 2*np.pi
        
        # Производные ошибок (простая аппроксимация)
        de_x = (e_x - self.prev_error[0]) / dt if dt > 0 else 0
        de_y = (e_y - self.prev_error[1]) / dt if dt > 0 else 0
        
        # Обновление интегральных ошибок
        self.integral_error[0] += e_x * dt
        self.integral_error[1] += e_y * dt
        
        # Ограничение интегральных ошибок
        self.integral_error = np.clip(self.integral_error, -10, 10)
        
        # Матрица поворота
        R = self.rotation_matrix(theta)
        
        # Матрица Σ(βs1, βs2)
        Sigma = self.sigma_matrix(beta_s1, beta_s2)
        
        # Желаемые ускорения
        xi_ref_ddot = np.array([ax_ref, ay_ref, alpha_ref])
        
        # Улучшенная динамическая линеаризация
        try:
            R_Sigma = R.T @ Sigma
            R_Sigma_pinv = np.linalg.pinv(R_Sigma)
            
            # Адаптивные коэффициенты
            kp_adaptive = self.kp1 * (1 + 0.05 * np.sqrt(e_x**2 + e_y**2))
            kd_adaptive = self.kd1 * (1 + 0.02 * np.sqrt(de_x**2 + de_y**2))
            
            # ПИД-управление
            pid_term = np.array([
                kp_adaptive * e_x + self.ki1 * self.integral_error[0] + kd_adaptive * de_x,
                kp_adaptive * e_y + self.ki2 * self.integral_error[1] + kd_adaptive * de_y,
                3.0 * e_theta
            ])
            
            # Закон управления
            eta = R_Sigma_pinv @ (xi_ref_ddot + pid_term)
            
        except np.linalg.LinAlgError:
            eta = np.array([ax_ref, ay_ref])
        
        # Улучшенный ПИД для углов рулевых колес
        zeta = np.array([
            self.kp1 * e_theta + self.ki1 * self.integral_error[0] + self.kd1 * de_x,
            self.kp2 * e_theta + self.ki2 * self.integral_error[1] + self.kd2 * de_y
        ])
        
        # Сохранение ошибок для следующего шага
        self.prev_error = np.array([e_x, e_y])
        
        return eta, zeta

def generate_reference_trajectory():
    """Генерация опорной траектории согласно изображению"""
    
    # Параметры траектории
    R1 = 7.0
    R2 = 3.0
    delta = 2*np.pi
    alpha = np.pi/6
    t_straight = 6.0
    v = 1.0
    
    # Времена сегментов
    t1 = R1 * delta / v
    t2 = t_straight
    t3 = R2 * np.pi / v
    t_total = t1 + t2 + t3
    
    # Массив времени
    dt = 0.1
    t_ref = np.arange(0, t_total + dt, dt)
    
    # Инициализация
    x_ref = np.zeros_like(t_ref)
    y_ref = np.zeros_like(t_ref)
    theta_ref = np.zeros_like(t_ref)
    vx_ref = np.zeros_like(t_ref)
    vy_ref = np.zeros_like(t_ref)
    omega_ref = np.zeros_like(t_ref)
    ax_ref = np.zeros_like(t_ref)
    ay_ref = np.zeros_like(t_ref)
    alpha_ref = np.zeros_like(t_ref)
    
    # Сегмент 1: Окружность R1
    mask1 = t_ref <= t1
    t1_local = t_ref[mask1]
    
    xc1, yc1 = 0, R1
    phi1 = np.pi/2 + v * t1_local / R1
    
    x_ref[mask1] = xc1 + R1 * np.cos(phi1)
    y_ref[mask1] = yc1 + R1 * np.sin(phi1)
    theta_ref[mask1] = phi1 + np.pi/2
    vx_ref[mask1] = -v * np.sin(phi1)
    vy_ref[mask1] = v * np.cos(phi1)
    omega_ref[mask1] = v / R1
    ax_ref[mask1] = -v**2/R1 * np.cos(phi1)
    ay_ref[mask1] = -v**2/R1 * np.sin(phi1)
    
    # Сегмент 2: Прямая
    mask2 = (t_ref > t1) & (t_ref <= t1 + t2)
    t2_local = t_ref[mask2] - t1
    
    phi1_end = np.pi/2 + v * t1 / R1
    x_end1 = xc1 + R1 * np.cos(phi1_end)
    y_end1 = yc1 + R1 * np.sin(phi1_end)
    theta_end1 = phi1_end + np.pi/2
    theta_turn = theta_end1 + alpha
    
    x_ref[mask2] = x_end1 + v * t2_local * np.cos(theta_turn)
    y_ref[mask2] = y_end1 + v * t2_local * np.sin(theta_turn)
    theta_ref[mask2] = theta_turn
    vx_ref[mask2] = v * np.cos(theta_turn)
    vy_ref[mask2] = v * np.sin(theta_turn)
    omega_ref[mask2] = 0
    ax_ref[mask2] = 0
    ay_ref[mask2] = 0
    
    # Сегмент 3: Окружность R2
    mask3 = t_ref > t1 + t2
    t3_local = t_ref[mask3] - t1 - t2
    
    x_start2 = x_end1 + v * t2 * np.cos(theta_turn)
    y_start2 = y_end1 + v * t2 * np.sin(theta_turn)
    
    center_offset = R2 * np.array([-np.sin(theta_turn), np.cos(theta_turn)])
    xc2 = x_start2 + center_offset[0]
    yc2 = y_start2 + center_offset[1]
    
    phi2_start = np.arctan2(y_start2 - yc2, x_start2 - xc2)
    phi2 = phi2_start - v * t3_local / R2
    
    x_ref[mask3] = xc2 + R2 * np.cos(phi2)
    y_ref[mask3] = yc2 + R2 * np.sin(phi2)
    theta_ref[mask3] = phi2 - np.pi/2
    vx_ref[mask3] = -v * np.sin(phi2)
    vy_ref[mask3] = v * np.cos(phi2)
    omega_ref[mask3] = -v / R2
    ax_ref[mask3] = v**2/R2 * np.cos(phi2)
    ay_ref[mask3] = v**2/R2 * np.sin(phi2)
    
    return t_ref, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref, ax_ref, ay_ref, alpha_ref

def simulate_improved_controllers():
    """Симуляция с улучшенными контроллерами"""
    
    # Создание контроллеров
    static_controller = ImprovedThreeWheelController()
    dynamic_controller = ImprovedThreeWheelController()
    
    # Генерация опорной траектории
    t_ref, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref, ax_ref, ay_ref, alpha_ref = generate_reference_trajectory()
    
    # Интерполяция
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
    initial_state = np.array([0.0, 7.0, np.pi/2, 0.0, 0.0])  # [x, y, theta, beta_s1, beta_s2]
    
    # Параметры симуляции
    dt = 0.05  # Уменьшенный шаг для лучшей точности
    t_sim = np.arange(0, 60, dt)
    
    # Симуляция со статической линеаризацией
    states_static = np.zeros((len(t_sim), 5))
    states_static[0] = initial_state
    
    for i in range(1, len(t_sim)):
        current_state = states_static[i-1]
        
        ref_state = np.array([x_ref_func(t_sim[i]), y_ref_func(t_sim[i]), theta_ref_func(t_sim[i])])
        ref_vel = np.array([vx_ref_func(t_sim[i]), vy_ref_func(t_sim[i]), omega_ref_func(t_sim[i])])
        
        eta, zeta = static_controller.improved_static_controller(current_state, ref_state, ref_vel)
        
        def ode_func(state, t):
            return static_controller.dynamics(state, t, eta, zeta)
        
        state_dot = ode_func(current_state, t_sim[i])
        states_static[i] = current_state + state_dot * dt
    
    # Симуляция с динамической линеаризацией
    states_dynamic = np.zeros((len(t_sim), 5))
    states_dynamic[0] = initial_state
    
    for i in range(1, len(t_sim)):
        current_state = states_dynamic[i-1]
        
        ref_state = np.array([x_ref_func(t_sim[i]), y_ref_func(t_sim[i]), theta_ref_func(t_sim[i])])
        ref_vel = np.array([vx_ref_func(t_sim[i]), vy_ref_func(t_sim[i]), omega_ref_func(t_sim[i])])
        ref_acc = np.array([ax_ref_func(t_sim[i]), ay_ref_func(t_sim[i]), alpha_ref_func(t_sim[i])])
        
        eta, zeta = dynamic_controller.improved_dynamic_controller(current_state, ref_state, ref_vel, ref_acc, dt)
        
        def ode_func(state, t):
            return dynamic_controller.dynamics(state, t, eta, zeta)
        
        state_dot = ode_func(current_state, t_sim[i])
        states_dynamic[i] = current_state + state_dot * dt
    
    return t_sim, states_static, states_dynamic, t_ref, x_ref, y_ref, theta_ref

def plot_improved_comparison():
    """Построение сравнения улучшенных контроллеров"""
    
    # Симуляция
    t_sim, states_static, states_dynamic, t_ref, x_ref, y_ref, theta_ref = simulate_improved_controllers()
    
    # Создание фигуры
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График траекторий
    ax1 = axes[0, 0]
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, label='Опорная траектория')
    ax1.plot(states_static[:, 0], states_static[:, 1], 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax1.plot(states_dynamic[:, 0], states_dynamic[:, 1], 'r-', linewidth=1.5, label='Динамическая линеаризация')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектории движения с улучшенными контроллерами')
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
    
    # График общей ошибки
    ax4 = axes[1, 1]
    error_norm_static = np.sqrt(error_x_static**2 + error_y_static**2)
    error_norm_dynamic = np.sqrt(error_x_dynamic**2 + error_y_dynamic**2)
    ax4.plot(t_sim, error_norm_static, 'b-', label='Статическая линеаризация')
    ax4.plot(t_sim, error_norm_dynamic, 'r-', label='Динамическая линеаризация')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Норма ошибки (м)')
    ax4.set_title('Общая ошибка слежения')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task2', exist_ok=True)
    plt.savefig('../images/task2/controllers_comparison_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Расчет статистики ошибок
    print("📊 Статистика ошибок с улучшенными контроллерами:")
    print(f"Статическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_static):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_static):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_static):.4f} м")
    
    print(f"Динамическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_dynamic):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_dynamic):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_dynamic):.4f} м")
    
    print("✅ Сравнение улучшенных контроллеров сохранено: controllers_comparison_improved.png")

if __name__ == "__main__":
    print("🚀 Симуляция улучшенных контроллеров для трехколесного робота")
    plot_improved_comparison()
