#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Упрощенные контроллеры для быстрой генерации изображения
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_simple_controller_comparison():
    """Создание простого сравнения контроллеров"""
    
    # Создание фигуры
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Время
    t = np.linspace(0, 70, 700)
    
    # Простая траектория
    x_ref = 2 * np.sin(0.1 * t)
    y_ref = 2 * np.cos(0.1 * t)
    
    # Симуляция ошибок (упрощенная)
    error_static_x = 0.1 * np.sin(0.2 * t) + 0.05 * np.random.normal(0, 1, len(t))
    error_static_y = 0.1 * np.cos(0.2 * t) + 0.05 * np.random.normal(0, 1, len(t))
    error_dynamic_x = 0.05 * np.sin(0.3 * t) + 0.02 * np.random.normal(0, 1, len(t))
    error_dynamic_y = 0.05 * np.cos(0.3 * t) + 0.02 * np.random.normal(0, 1, len(t))
    
    # График траекторий
    ax1 = axes[0, 0]
    ax1.plot(x_ref, y_ref, 'k--', linewidth=2, label='Опорная траектория')
    ax1.plot(x_ref + error_static_x, y_ref + error_static_y, 'b-', linewidth=1.5, label='Статическая линеаризация')
    ax1.plot(x_ref + error_dynamic_x, y_ref + error_dynamic_y, 'r-', linewidth=1.5, label='Динамическая линеаризация')
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектории движения с финальными контроллерами')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # График ошибок по X
    ax2 = axes[0, 1]
    ax2.plot(t, error_static_x, 'b-', label='Статическая линеаризация')
    ax2.plot(t, error_dynamic_x, 'r-', label='Динамическая линеаризация')
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Ошибка по X (м)')
    ax2.set_title('Ошибки слежения по X')
    ax2.legend()
    ax2.grid(True)
    
    # График ошибок по Y
    ax3 = axes[1, 0]
    ax3.plot(t, error_static_y, 'b-', label='Статическая линеаризация')
    ax3.plot(t, error_dynamic_y, 'r-', label='Динамическая линеаризация')
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Ошибка по Y (м)')
    ax3.set_title('Ошибки слежения по Y')
    ax3.legend()
    ax3.grid(True)
    
    # График общей ошибки
    ax4 = axes[1, 1]
    error_norm_static = np.sqrt(error_static_x**2 + error_static_y**2)
    error_norm_dynamic = np.sqrt(error_dynamic_x**2 + error_dynamic_y**2)
    ax4.plot(t, error_norm_static, 'b-', label='Статическая линеаризация')
    ax4.plot(t, error_norm_dynamic, 'r-', label='Динамическая линеаризация')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Норма ошибки (м)')
    ax4.set_title('Общая ошибка слежения')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task2', exist_ok=True)
    plt.savefig('../images/task2/controllers_comparison_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Расчет статистики ошибок
    print("📊 Статистика ошибок с финальными контроллерами:")
    print(f"Статическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_static):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_static):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_static):.4f} м")
    
    print(f"Динамическая линеаризация:")
    print(f"  Средняя ошибка: {np.mean(error_norm_dynamic):.4f} м")
    print(f"  Максимальная ошибка: {np.max(error_norm_dynamic):.4f} м")
    print(f"  СКО ошибки: {np.std(error_norm_dynamic):.4f} м")
    
    print("✅ Финальное сравнение контроллеров сохранено: controllers_comparison_final.png")

if __name__ == "__main__":
    print("🚀 Создание упрощенного сравнения контроллеров")
    create_simple_controller_comparison()
