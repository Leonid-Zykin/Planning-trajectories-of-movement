#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Создание диаграммы трехколесного мобильного робота типа (1,2)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Настройка matplotlib для русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_three_wheel_robot_diagram():
    """Создание диаграммы трехколесного робота"""
    
    # Создание фигуры
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Параметры робота
    L = 0.3  # База робота
    W = 0.2  # Колея
    R = 0.05  # Радиус колес
    d = 0.1   # Геометрический параметр
    
    # Координаты центра масс
    center_x, center_y = 0, 0
    
    # Координаты колес
    # Рулевые колеса (передние)
    wheel1_x = center_x + L/2
    wheel1_y = center_y + W/2
    wheel2_x = center_x + L/2
    wheel2_y = center_y - W/2
    
    # Направляющее колесо (заднее)
    wheel3_x = center_x - L/2
    wheel3_y = center_y
    
    # Корпус робота (треугольник)
    body_points = np.array([
        [center_x + L/2, center_y + W/2],  # Передний правый угол
        [center_x + L/2, center_y - W/2],  # Передний левый угол
        [center_x - L/2, center_y]         # Задний центр
    ])
    
    # Рисование корпуса
    body = patches.Polygon(body_points, closed=True, facecolor='lightblue', 
                          edgecolor='blue', linewidth=2, alpha=0.7)
    ax.add_patch(body)
    
    # Рисование колес
    wheel1 = patches.Circle((wheel1_x, wheel1_y), R, facecolor='gray', 
                           edgecolor='black', linewidth=2)
    wheel2 = patches.Circle((wheel2_x, wheel2_y), R, facecolor='gray', 
                           edgecolor='black', linewidth=2)
    wheel3 = patches.Circle((wheel3_x, wheel3_y), R, facecolor='gray', 
                           edgecolor='black', linewidth=2)
    
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    ax.add_patch(wheel3)
    
    # Рисование осей колес
    ax.plot([wheel1_x - R, wheel1_x + R], [wheel1_y, wheel1_y], 'k-', linewidth=3)
    ax.plot([wheel2_x - R, wheel2_x + R], [wheel2_y, wheel2_y], 'k-', linewidth=3)
    ax.plot([wheel3_x - R, wheel3_x + R], [wheel3_y, wheel3_y], 'k-', linewidth=3)
    
    # Рисование стрелок для рулевых колес
    arrow_length = 0.08
    ax.arrow(wheel1_x, wheel1_y, arrow_length, 0, head_width=0.02, head_length=0.02, 
             fc='red', ec='red', linewidth=2)
    ax.arrow(wheel2_x, wheel2_y, arrow_length, 0, head_width=0.02, head_length=0.02, 
             fc='red', ec='red', linewidth=2)
    
    # Центр масс
    ax.plot(center_x, center_y, 'ko', markersize=8, label='Центр масс')
    
    # Система координат
    ax.arrow(center_x, center_y, 0.15, 0, head_width=0.02, head_length=0.02, 
             fc='green', ec='green', linewidth=2)
    ax.arrow(center_x, center_y, 0, 0.15, head_width=0.02, head_length=0.02, 
             fc='green', ec='green', linewidth=2)
    ax.text(center_x + 0.16, center_y, 'X', fontsize=12, color='green', fontweight='bold')
    ax.text(center_x, center_y + 0.16, 'Y', fontsize=12, color='green', fontweight='bold')
    
    # Подписи параметров
    ax.text(center_x + L/4, center_y + W/2 + 0.05, f'L = {L} м', fontsize=10, ha='center')
    ax.text(center_x + L/2 + 0.05, center_y, f'W = {W} м', fontsize=10, va='center')
    ax.text(wheel1_x + 0.05, wheel1_y + 0.05, f'R = {R} м', fontsize=10)
    
    # Подписи колес
    ax.text(wheel1_x, wheel1_y + 0.08, 'Рулевое\nколесо 1', fontsize=9, ha='center')
    ax.text(wheel2_x, wheel2_y - 0.08, 'Рулевое\nколесо 2', fontsize=9, ha='center')
    ax.text(wheel3_x, wheel3_y - 0.08, 'Направляющее\nколесо', fontsize=9, ha='center')
    
    # Углы поворота рулевых колес
    beta1_angle = np.pi/6  # 30 градусов
    beta2_angle = -np.pi/6  # -30 градусов
    
    # Стрелки для углов поворота
    arrow_len = 0.06
    ax.arrow(wheel1_x, wheel1_y, arrow_len*np.cos(beta1_angle), arrow_len*np.sin(beta1_angle), 
             head_width=0.015, head_length=0.015, fc='orange', ec='orange', linewidth=2)
    ax.arrow(wheel2_x, wheel2_y, arrow_len*np.cos(beta2_angle), arrow_len*np.sin(beta2_angle), 
             head_width=0.015, head_length=0.015, fc='orange', ec='orange', linewidth=2)
    
    ax.text(wheel1_x + 0.08, wheel1_y + 0.05, 'βs1', fontsize=10, color='orange', fontweight='bold')
    ax.text(wheel2_x + 0.08, wheel2_y - 0.05, 'βs2', fontsize=10, color='orange', fontweight='bold')
    
    # Настройка осей
    ax.set_xlim(center_x - L/2 - 0.2, center_x + L/2 + 0.2)
    ax.set_ylim(center_y - W/2 - 0.2, center_y + W/2 + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_title('Схема трехколесного мобильного робота типа (1,2)', fontsize=14, fontweight='bold')
    
    # Легенда
    legend_elements = [
        patches.Patch(color='lightblue', label='Корпус робота'),
        patches.Patch(color='gray', label='Колеса'),
        plt.Line2D([0], [0], color='red', linewidth=2, label='Рулевые колеса'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Углы поворота βs1, βs2'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Система координат'),
        plt.Line2D([0], [0], marker='o', color='black', linewidth=0, markersize=8, label='Центр масс')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Добавление информации о типе робота
    info_text = """Тип робота (1,2):
• Два рулевых колеса (передние)
• Одно направляющее колесо (заднее)
• Степень мобильности: δm = 2
• Степень управляемости: δs = 2"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Сохранение
    os.makedirs('../images/task1', exist_ok=True)
    plt.savefig('../images/task1/robot_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Диаграмма трехколесного робота сохранена: robot_diagram.png")

if __name__ == "__main__":
    print("🚀 Создание диаграммы трехколесного мобильного робота")
    create_three_wheel_robot_diagram()
