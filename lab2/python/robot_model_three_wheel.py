#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модель трехколесного мобильного робота типа (1,2)
Согласно методичке: два рулевых колеса + одно направляющее колесо
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ThreeWheelMobileRobot:
    """Трехколесный мобильный робот типа (1,2)"""
    
    def __init__(self):
        # Геометрические параметры
        self.L = 0.3  # База робота (расстояние от центра масс до колес)
        self.W = 0.2  # Колея (расстояние между рулевыми колесами)
        self.R = 0.05  # Радиус колес
        self.d = 0.1   # Геометрический параметр
        
        # Физические параметры
        self.m = 10.0  # Масса робота
        self.I = 1.0   # Момент инерции
        
        # Матрица конфигурации приводов для трехколесного робота
        self.B = np.array([
            [1, 1, 0],
            [0, 0, 1], 
            [1, -1, 0]
        ]) / (2 * self.R)
        
        # Матрица поворота
        self.R_matrix = None
        
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
    
    def simulate(self, initial_state, t_span, eta_func, zeta_func):
        """
        Симуляция движения робота
        
        Args:
            initial_state: начальное состояние [x, y, theta, beta_s1, beta_s2]
            t_span: массив времени
            eta_func: функция обобщенных скоростей eta(t)
            zeta_func: функция скоростей изменения углов zeta(t)
        
        Returns:
            t: массив времени
            states: массив состояний
        """
        def ode_func(state, t):
            eta = eta_func(t)
            zeta = zeta_func(t)
            return self.dynamics(state, t, eta, zeta)
        
        states = odeint(ode_func, initial_state, t_span)
        return t_span, states

def generate_trajectory_variant5():
    """Генерация траектории для варианта 5"""
    
    # Параметры траектории (вариант 5)
    R1 = 7.0      # Радиус первой окружности
    R2 = 3.0      # Радиус второй окружности
    delta = 2*np.pi  # Угол поворота на первой окружности
    alpha = np.pi/6  # Угол поворота перед движением по прямой
    t_straight = 6.0  # Время движения по прямой
    
    # Скорость движения
    v = 1.0  # м/с
    
    # Расчет времени для каждого участка
    t1 = R1 * delta / v  # Время движения по первой окружности
    t2 = t_straight      # Время движения по прямой
    t3 = R2 * np.pi / v  # Время движения по второй окружности
    
    # Общее время
    t_total = t1 + t2 + t3
    
    # Создание массива времени
    dt = 0.1
    t = np.arange(0, t_total + dt, dt)
    
    # Инициализация массивов
    x_ref = np.zeros_like(t)
    y_ref = np.zeros_like(t)
    theta_ref = np.zeros_like(t)
    vx_ref = np.zeros_like(t)
    vy_ref = np.zeros_like(t)
    omega_ref = np.zeros_like(t)
    
    # Участок 1: Движение по окружности R1
    mask1 = t <= t1
    t1_local = t[mask1]
    
    x_ref[mask1] = R1 * np.sin(v * t1_local / R1)
    y_ref[mask1] = R1 * (1 - np.cos(v * t1_local / R1))
    theta_ref[mask1] = v * t1_local / R1
    vx_ref[mask1] = v * np.cos(theta_ref[mask1])
    vy_ref[mask1] = v * np.sin(theta_ref[mask1])
    omega_ref[mask1] = v / R1
    
    # Участок 2: Движение по прямой
    mask2 = (t > t1) & (t <= t1 + t2)
    t2_local = t[mask2] - t1
    
    # Начальная позиция для прямого участка
    x_start = R1 * np.sin(delta)
    y_start = R1 * (1 - np.cos(delta))
    theta_start = delta
    
    # Поворот на угол alpha
    theta_turn = theta_start + alpha
    
    x_ref[mask2] = x_start + v * t2_local * np.cos(theta_turn)
    y_ref[mask2] = y_start + v * t2_local * np.sin(theta_turn)
    theta_ref[mask2] = theta_turn
    vx_ref[mask2] = v * np.cos(theta_turn)
    vy_ref[mask2] = v * np.sin(theta_turn)
    omega_ref[mask2] = 0
    
    # Участок 3: Движение по окружности R2
    mask3 = t > t1 + t2
    t3_local = t[mask3] - t1 - t2
    
    # Начальная позиция для второй окружности
    x_start2 = x_start + v * t2 * np.cos(theta_turn)
    y_start2 = y_start + v * t2 * np.sin(theta_turn)
    
    # Центр второй окружности
    xc2 = x_start2 - R2 * np.sin(theta_turn)
    yc2 = y_start2 + R2 * np.cos(theta_turn)
    
    # Угол для второй окружности
    phi2 = theta_turn + np.pi/2 + v * t3_local / R2
    
    x_ref[mask3] = xc2 + R2 * np.cos(phi2)
    y_ref[mask3] = yc2 + R2 * np.sin(phi2)
    theta_ref[mask3] = phi2 - np.pi/2
    vx_ref[mask3] = v * np.cos(theta_ref[mask3])
    vy_ref[mask3] = v * np.sin(theta_ref[mask3])
    omega_ref[mask3] = v / R2
    
    return t, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref

def plot_trajectory_variant5():
    """Построение траектории для варианта 5"""
    
    # Генерация траектории
    t, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref = generate_trajectory_variant5()
    
    # Создание фигуры
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График траектории
    ax1.plot(x_ref, y_ref, 'b-', linewidth=2, label='Траектория')
    
    # Отметки участков
    # Участок 1: окружность R1
    mask1 = t <= 7.0 * 2 * np.pi / 1.0
    ax1.plot(x_ref[mask1], y_ref[mask1], 'r-', linewidth=3, alpha=0.7, label='Участок 1: Окружность R₁=7м')
    
    # Участок 2: прямая
    mask2 = (t > 7.0 * 2 * np.pi / 1.0) & (t <= 7.0 * 2 * np.pi / 1.0 + 6.0)
    ax1.plot(x_ref[mask2], y_ref[mask2], 'g-', linewidth=3, alpha=0.7, label='Участок 2: Прямая')
    
    # Участок 3: окружность R2
    mask3 = t > 7.0 * 2 * np.pi / 1.0 + 6.0
    ax1.plot(x_ref[mask3], y_ref[mask3], 'm-', linewidth=3, alpha=0.7, label='Участок 3: Окружность R₂=3м')
    
    # Начальная точка
    ax1.plot(x_ref[0], y_ref[0], 'go', markersize=10, label='Начальная точка')
    
    # Конечная точка
    ax1.plot(x_ref[-1], y_ref[-1], 'ro', markersize=10, label='Конечная точка')
    
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектория движения трехколесного мобильного робота (вариант 5)')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # График скоростей
    ax2.plot(t, vx_ref, 'b-', label='Vx (м/с)')
    ax2.plot(t, vy_ref, 'r-', label='Vy (м/с)')
    ax2.plot(t, omega_ref, 'g-', label='ω (рад/с)')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Скорость')
    ax2.set_title('Скорости движения')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task1', exist_ok=True)
    plt.savefig('../images/task1/trajectory_variant5_three_wheel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Траектория для трехколесного робота сохранена: trajectory_variant5_three_wheel.png")

if __name__ == "__main__":
    print("🚀 Генерация траектории для трехколесного мобильного робота (вариант 5)")
    plot_trajectory_variant5()
