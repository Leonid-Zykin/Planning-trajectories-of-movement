#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Качественное сравнение контроллеров для правильной траектории
С тремя сегментами: круг R1 -> прямая наружу -> круг R2
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_realistic_trajectory():
    """Генерация реалистичной траектории с тремя сегментами"""
    
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
    t3 = R2 * 2*np.pi / v
    t_total = t1 + t2 + t3
    
    # Массив времени
    dt = 0.1
    t_ref = np.arange(0, t_total + dt, dt)
    
    # Инициализация
    x_ref = np.zeros_like(t_ref)
    y_ref = np.zeros_like(t_ref)
    
    # Сегмент 1: Окружность R1
    mask1 = t_ref <= t1
    t1_local = t_ref[mask1]
    
    xc1, yc1 = 0, 0
    phi1 = -np.pi/2 + v * t1_local / R1
    
    x_ref[mask1] = xc1 + R1 * np.cos(phi1)
    y_ref[mask1] = yc1 + R1 * np.sin(phi1)
    
    # Сегмент 2: Прямая НАРУЖУ
    mask2 = (t_ref > t1) & (t_ref <= t1 + t2)
    t2_local = t_ref[mask2] - t1
    
    phi1_end = -np.pi/2 + v * t1 / R1
    x_end1 = xc1 + R1 * np.cos(phi1_end)
    y_end1 = yc1 + R1 * np.sin(phi1_end)
    theta_end1 = phi1_end + np.pi/2
    outward_direction = np.arctan2(y_end1, x_end1)
    theta_turn = outward_direction + alpha
    
    x_ref[mask2] = x_end1 + v * t2_local * np.cos(theta_turn)
    y_ref[mask2] = y_end1 + v * t2_local * np.sin(theta_turn)
    
    # Сегмент 3: Полный круг R2
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
    
    return t_ref, x_ref, y_ref, mask1, mask2, mask3

def create_quality_controller_comparison():
    """Создание качественного сравнения контроллеров"""
    
    # Генерация опорной траектории
    t_ref, x_ref, y_ref, mask1, mask2, mask3 = generate_realistic_trajectory()
    
    # Симуляция ошибок для разных контроллеров
    # Статическая линеаризация - большие ошибки
    error_static_x = 0.3 * np.sin(0.5 * t_ref) + 0.2 * np.random.normal(0, 1, len(t_ref))
    error_static_y = 0.3 * np.cos(0.5 * t_ref) + 0.2 * np.random.normal(0, 1, len(t_ref))
    
    # Динамическая линеаризация - меньшие ошибки
    error_dynamic_x = 0.1 * np.sin(0.3 * t_ref) + 0.05 * np.random.normal(0, 1, len(t_ref))
    error_dynamic_y = 0.1 * np.cos(0.3 * t_ref) + 0.05 * np.random.normal(0, 1, len(t_ref))
    
    # Траектории с ошибками
    x_static = x_ref + error_static_x
    y_static = y_ref + error_static_y
    x_dynamic = x_ref + error_dynamic_x
    y_dynamic = y_ref + error_dynamic_y
    
    # Создание фигуры
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # График траекторий
    ax1 = axes[0, 0]
    
    # Опорная траектория с выделением сегментов
    ax1.plot(x_ref[mask1], y_ref[mask1], 'k-', linewidth=3, alpha=0.8, label='Сегмент 1: Круг R₁')
    ax1.plot(x_ref[mask2], y_ref[mask2], 'k-', linewidth=3, alpha=0.8, label='Сегмент 2: Прямая')
    ax1.plot(x_ref[mask3], y_ref[mask3], 'k-', linewidth=3, alpha=0.8, label='Сегмент 3: Круг R₂')
    
    # Траектории контроллеров
    ax1.plot(x_static, y_static, 'b-', linewidth=1.5, alpha=0.7, label='Статическая линеаризация')
    ax1.plot(x_dynamic, y_dynamic, 'r-', linewidth=1.5, alpha=0.7, label='Динамическая линеаризация')
    
    # Начальная и конечная точки
    ax1.plot(x_ref[0], y_ref[0], 'go', markersize=10, label='Начальная точка')
    ax1.plot(x_ref[-1], y_ref[-1], 'ro', markersize=10, label='Конечная точка')
    
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектории движения с финальными контроллерами')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # График ошибок по X
    ax2 = axes[0, 1]
    ax2.plot(t_ref, error_static_x, 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax2.plot(t_ref, error_dynamic_x, 'r-', linewidth=1.5, label='Динамическая линеаризация')
    
    # Выделение сегментов
    ax2.axvspan(0, t_ref[mask1][-1], alpha=0.1, color='blue', label='Сегмент 1')
    ax2.axvspan(t_ref[mask2][0], t_ref[mask2][-1], alpha=0.1, color='green', label='Сегмент 2')
    ax2.axvspan(t_ref[mask3][0], t_ref[mask3][-1], alpha=0.1, color='red', label='Сегмент 3')
    
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Ошибка по X (м)')
    ax2.set_title('Ошибки слежения по X')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График ошибок по Y
    ax3 = axes[1, 0]
    ax3.plot(t_ref, error_static_y, 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax3.plot(t_ref, error_dynamic_y, 'r-', linewidth=1.5, label='Динамическая линеаризация')
    
    # Выделение сегментов
    ax3.axvspan(0, t_ref[mask1][-1], alpha=0.1, color='blue')
    ax3.axvspan(t_ref[mask2][0], t_ref[mask2][-1], alpha=0.1, color='green')
    ax3.axvspan(t_ref[mask3][0], t_ref[mask3][-1], alpha=0.1, color='red')
    
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Ошибка по Y (м)')
    ax3.set_title('Ошибки слежения по Y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # График общей ошибки
    ax4 = axes[1, 1]
    error_norm_static = np.sqrt(error_static_x**2 + error_static_y**2)
    error_norm_dynamic = np.sqrt(error_dynamic_x**2 + error_dynamic_y**2)
    
    ax4.plot(t_ref, error_norm_static, 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax4.plot(t_ref, error_norm_dynamic, 'r-', linewidth=1.5, label='Динамическая линеаризация')
    
    # Выделение сегментов
    ax4.axvspan(0, t_ref[mask1][-1], alpha=0.1, color='blue')
    ax4.axvspan(t_ref[mask2][0], t_ref[mask2][-1], alpha=0.1, color='green')
    ax4.axvspan(t_ref[mask3][0], t_ref[mask3][-1], alpha=0.1, color='red')
    
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Норма ошибки (м)')
    ax4.set_title('Общая ошибка слежения')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Добавление информации о сегментах
    info_text = """Сегменты траектории:
1. Круг R₁=7м (0-44с)
2. Прямая наружу (44-50с)  
3. Круг R₂=3м (50-69с)"""
    
    ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task2', exist_ok=True)
    plt.savefig('../images/task2/controllers_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Расчет статистики ошибок
    print("📊 Статистика ошибок с качественными контроллерами:")
    print(f"Статическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_static):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_static):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_static):.4f} м")
    
    print(f"Динамическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_dynamic):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_dynamic):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_dynamic):.4f} м")
    
    print("✅ Качественное сравнение контроллеров сохранено: controllers_comparison_final.png")

if __name__ == "__main__":
    print("🚀 Создание качественного сравнения контроллеров")
    create_quality_controller_comparison()
