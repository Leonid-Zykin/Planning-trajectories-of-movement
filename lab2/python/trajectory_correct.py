#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Правильная траектория для варианта 5 согласно изображению
3 сегмента: окружность R1 -> поворот + прямая -> окружность R2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_correct_trajectory_variant5():
    """Генерация правильной траектории согласно изображению"""
    
    # Параметры траектории (вариант 5)
    R1 = 7.0      # Радиус первой окружности
    R2 = 3.0      # Радиус второй окружности  
    delta = 2*np.pi  # Угол поворота на первой окружности (полный оборот)
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
    
    # СЕГМЕНТ 1: Движение по окружности R1 (против часовой стрелки)
    mask1 = t <= t1
    t1_local = t[mask1]
    
    # Центр первой окружности в (0, R1)
    xc1, yc1 = 0, R1
    
    # Угол для первой окружности (против часовой стрелки)
    phi1 = np.pi/2 + v * t1_local / R1  # Начинаем с верхней точки
    
    x_ref[mask1] = xc1 + R1 * np.cos(phi1)
    y_ref[mask1] = yc1 + R1 * np.sin(phi1)
    theta_ref[mask1] = phi1 + np.pi/2  # Тангенциальное направление
    vx_ref[mask1] = -v * np.sin(phi1)
    vy_ref[mask1] = v * np.cos(phi1)
    omega_ref[mask1] = v / R1
    
    # СЕГМЕНТ 2: Поворот на α и движение по прямой
    mask2 = (t > t1) & (t <= t1 + t2)
    t2_local = t[mask2] - t1
    
    # Конечная точка первого сегмента
    phi1_end = np.pi/2 + v * t1 / R1
    x_end1 = xc1 + R1 * np.cos(phi1_end)
    y_end1 = yc1 + R1 * np.sin(phi1_end)
    theta_end1 = phi1_end + np.pi/2
    
    # Поворот на угол α
    theta_turn = theta_end1 + alpha
    
    # Движение по прямой
    x_ref[mask2] = x_end1 + v * t2_local * np.cos(theta_turn)
    y_ref[mask2] = y_end1 + v * t2_local * np.sin(theta_turn)
    theta_ref[mask2] = theta_turn
    vx_ref[mask2] = v * np.cos(theta_turn)
    vy_ref[mask2] = v * np.sin(theta_turn)
    omega_ref[mask2] = 0
    
    # СЕГМЕНТ 3: Движение по окружности R2 (по часовой стрелке)
    mask3 = t > t1 + t2
    t3_local = t[mask3] - t1 - t2
    
    # Начальная точка для второй окружности
    x_start2 = x_end1 + v * t2 * np.cos(theta_turn)
    y_start2 = y_end1 + v * t2 * np.sin(theta_turn)
    
    # Центр второй окружности (расположен так, чтобы касаться прямой)
    # Направление от центра к точке касания
    tangent_dir = np.array([np.cos(theta_turn), np.sin(theta_turn)])
    center_offset = R2 * np.array([-np.sin(theta_turn), np.cos(theta_turn)])
    
    xc2 = x_start2 + center_offset[0]
    yc2 = y_start2 + center_offset[1]
    
    # Угол для второй окружности (по часовой стрелке)
    phi2_start = np.arctan2(y_start2 - yc2, x_start2 - xc2)
    phi2 = phi2_start - v * t3_local / R2  # Минус для движения по часовой стрелке
    
    x_ref[mask3] = xc2 + R2 * np.cos(phi2)
    y_ref[mask3] = yc2 + R2 * np.sin(phi2)
    theta_ref[mask3] = phi2 - np.pi/2  # Тангенциальное направление
    vx_ref[mask3] = -v * np.sin(phi2)
    vy_ref[mask3] = v * np.cos(phi2)
    omega_ref[mask3] = -v / R2  # Минус для движения по часовой стрелке
    
    return t, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref

def plot_correct_trajectory_variant5():
    """Построение правильной траектории согласно изображению"""
    
    # Генерация траектории
    t, x_ref, y_ref, theta_ref, vx_ref, vy_ref, omega_ref = generate_correct_trajectory_variant5()
    
    # Создание фигуры
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # График траектории
    ax1.plot(x_ref, y_ref, 'k-', linewidth=2, label='Траектория')
    
    # Отметки участков
    # Сегмент 1: окружность R1
    mask1 = t <= 7.0 * 2 * np.pi / 1.0
    ax1.plot(x_ref[mask1], y_ref[mask1], 'r-', linewidth=3, alpha=0.8, label='Сегмент 1: Окружность R₁=7м')
    
    # Сегмент 2: прямая
    mask2 = (t > 7.0 * 2 * np.pi / 1.0) & (t <= 7.0 * 2 * np.pi / 1.0 + 6.0)
    ax1.plot(x_ref[mask2], y_ref[mask2], 'g-', linewidth=3, alpha=0.8, label='Сегмент 2: Прямая')
    
    # Сегмент 3: окружность R2
    mask3 = t > 7.0 * 2 * np.pi / 1.0 + 6.0
    ax1.plot(x_ref[mask3], y_ref[mask3], 'b-', linewidth=3, alpha=0.8, label='Сегмент 3: Окружность R₂=3м')
    
    # Начальная точка
    ax1.plot(x_ref[0], y_ref[0], 'go', markersize=10, label='Начальная точка')
    
    # Конечная точка
    ax1.plot(x_ref[-1], y_ref[-1], 'ro', markersize=10, label='Конечная точка')
    
    # Центры окружностей
    ax1.plot(0, 7, 'rx', markersize=8, label='Центр R₁')
    ax1.plot(x_ref[mask3][0] - 3*np.cos(np.arctan2(y_ref[mask3][0] - (y_ref[mask2][-1] + 3*np.sin(np.pi/6)), 
                                                   x_ref[mask3][0] - (x_ref[mask2][-1] + 3*np.cos(np.pi/6)))), 
             y_ref[mask3][0] - 3*np.sin(np.arctan2(y_ref[mask3][0] - (y_ref[mask2][-1] + 3*np.sin(np.pi/6)), 
                                                   x_ref[mask3][0] - (x_ref[mask2][-1] + 3*np.cos(np.pi/6)))), 
             'bx', markersize=8, label='Центр R₂')
    
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Правильная траектория движения мобильного робота (вариант 5)', fontsize=14)
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
    
    # Добавление информации о сегментах
    info_text = """Сегменты траектории:
1. Окружность R₁=7м, δ=2π рад (против часовой стрелки)
2. Поворот α=π/6 рад + прямая t=6с
3. Окружность R₂=3м (по часовой стрелке)"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task1', exist_ok=True)
    plt.savefig('../images/task1/trajectory_variant5_correct.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Правильная траектория сохранена: trajectory_variant5_correct.png")
    print(f"📊 Параметры траектории:")
    print(f"   Сегмент 1: R₁={7.0}м, δ={2*np.pi:.2f} рад, время={7.0*2*np.pi/1.0:.1f}с")
    print(f"   Сегмент 2: α={np.pi/6:.2f} рад, время={6.0}с")
    print(f"   Сегмент 3: R₂={3.0}м, время={3.0*np.pi/1.0:.1f}с")

if __name__ == "__main__":
    print("🚀 Генерация правильной траектории для варианта 5")
    plot_correct_trajectory_variant5()
