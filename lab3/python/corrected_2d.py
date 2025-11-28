import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os


class PathSegment:
    """Базовый класс параметрически заданного участка траектории."""

    def position(self, gamma):
        raise NotImplementedError

    def dp_dgamma(self, gamma):
        raise NotImplementedError

    def tangent(self, gamma):
        deriv = self.dp_dgamma(gamma)
        norm = np.linalg.norm(deriv)
        if norm < 1e-9:
            return np.array([1.0, 0.0])
        return deriv / norm

    def normal(self, gamma):
        tau = self.tangent(gamma)
        return np.array([-tau[1], tau[0]])

    def ds_dgamma(self, gamma):
        return np.linalg.norm(self.dp_dgamma(gamma))

    def length(self, samples=2000):
        gammas = np.linspace(0.0, 1.0, samples)
        ds = np.array([self.ds_dgamma(g) for g in gammas])
        return np.trapz(ds, gammas)


class CircleSegment(PathSegment):
    def __init__(self, radius=2.0, center=(0.0, 0.0)):
        self.radius = radius
        self.center = np.array(center)
        self._length = 2 * np.pi * self.radius

    def position(self, gamma):
        theta = 2 * np.pi * gamma
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])

    def dp_dgamma(self, gamma):
        theta = 2 * np.pi * gamma
        return 2 * np.pi * self.radius * np.array([-np.sin(theta), np.cos(theta)])

    def length(self, samples=0):
        return self._length


class EllipseSegment(PathSegment):
    def __init__(self, a=3.0, b=2.0, center=(2.0, 1.0)):
        self.a = a
        self.b = b
        self.center = np.array(center)
        self._length = super().length()

    def position(self, gamma):
        theta = 2 * np.pi * gamma
        return self.center + np.array([self.a * np.cos(theta), self.b * np.sin(theta)])

    def dp_dgamma(self, gamma):
        theta = 2 * np.pi * gamma
        return 2 * np.pi * np.array([-self.a * np.sin(theta), self.b * np.cos(theta)])

    def length(self, samples=2000):
        return super().length(samples)


class ParabolaSegment(PathSegment):
    def __init__(self, a=0.5, offset=-2.0, x_range=(-3.0, 3.0)):
        self.a = a
        self.offset = offset
        self.x0, self.x1 = x_range
        self._length = super().length()

    def _x(self, gamma):
        return self.x0 + (self.x1 - self.x0) * gamma

    def position(self, gamma):
        x = self._x(gamma)
        y = self.a * x**2 + self.offset
        return np.array([x, y])

    def dp_dgamma(self, gamma):
        dx_dgamma = (self.x1 - self.x0)
        x = self._x(gamma)
        dy_dx = 2 * self.a * x
        dy_dgamma = dy_dx * dx_dgamma
        return np.array([dx_dgamma, dy_dgamma])

    def length(self, samples=4000):
        return super().length(samples)


class CorrectedCoordinatedControl2D:
    """
    Улучшенный алгоритм стабилизации 2D-траекторий с параметризацией каждого участка.
    """

    def __init__(self, m=1.0, k=180.0, c=60.0, k_tau=15.0, d_tau=6.0,
                 adaptive_gain=0.2, u_limit=250.0, max_tangential_speed=5.0,
                 min_tangential_speed=0.2):
        self.m = m
        self.k = k
        self.c = c
        self.k_tau = k_tau
        self.d_tau = d_tau
        self.adaptive_gain = adaptive_gain
        self.u_limit = u_limit
        self.max_tangential_speed = max_tangential_speed
        self.min_tangential_speed = min_tangential_speed

        self.segments = [
            CircleSegment(radius=2.0, center=(0.0, 0.0)),
            EllipseSegment(a=3.0, b=2.0, center=(2.0, 1.0)),
            ParabolaSegment(a=0.5, offset=-2.0, x_range=(-3.0, 3.0))
        ]

        self.phi_funcs = [
            lambda x, y: x**2 + y**2 - 4,
            lambda x, y: (x-2)**2/9 + (y-1)**2/4 - 1,
            lambda x, y: y - 0.5*x**2 + 2
        ]

    def _reference_kinematics(self, segment, gamma, s_star, e_normal):
        tau = segment.tangent(gamma)
        n = segment.normal(gamma)
        s_effective = s_star * np.exp(-self.adaptive_gain * abs(e_normal))
        s_effective = np.clip(s_effective, self.min_tangential_speed, self.max_tangential_speed)
        ds_dgamma = max(segment.ds_dgamma(gamma), 1e-6)
        gamma_dot = s_effective / ds_dgamma
        return tau, n, s_effective, gamma_dot

    def control_law(self, state, segment, s_star):
        x, y, vx, vy, gamma = state
        pos = np.array([x, y])
        vel = np.array([vx, vy])
        ref = segment.position(gamma)
        error_vec = pos - ref

        tau, n, s_effective, gamma_dot = self._reference_kinematics(segment, gamma, s_star, np.dot(error_vec, segment.normal(gamma)))
        e_n = np.dot(error_vec, n)
        v_n = np.dot(vel, n)
        v_tau = np.dot(vel, tau)

        u_normal = -self.k * e_n * n - self.c * v_n * n
        u_tangential = self.k_tau * (s_effective - v_tau) * tau - self.d_tau * v_tau * tau
        u = u_normal + u_tangential
        u = np.clip(u, -self.u_limit, self.u_limit)

        return u, gamma_dot

    def dynamics(self, state, t, segment, s_star):
        u, gamma_dot = self.control_law(state, segment, s_star)
        x_dot = state[2]
        y_dot = state[3]
        vx_dot = u[0] / self.m
        vy_dot = u[1] / self.m
        return [x_dot, y_dot, vx_dot, vy_dot, gamma_dot]

    def simulate_segment(self, initial_state, segment, s_star, dt, max_time, phase_id):
        t_local = np.arange(0.0, max_time, dt)

        def rhs(state, t):
            return self.dynamics(state, t, segment, s_star)

        sol = odeint(rhs, initial_state, t_local, rtol=1e-7, atol=1e-9)
        gamma_vals = sol[:, 4]
        completed_idx = np.argmax(gamma_vals >= 0.999)
        if gamma_vals[completed_idx] < 0.999:
            completed_idx = len(t_local) - 1

        sol = sol[:completed_idx + 1]
        t_local = t_local[:completed_idx + 1]
        phase_vec = np.full(len(t_local), phase_id)
        return t_local, sol, phase_vec

    def simulate_trajectory(self, x0, t_grid, s_star):
        dt = t_grid[1] - t_grid[0]
        states = []
        times = []
        phases = []
        accumulated_time = 0.0

        current_state = np.hstack([x0, 0.0])  # gamma = 0

        for idx, segment in enumerate(self.segments, start=1):
            approx_time = segment.length() / max(s_star, 1e-3) * 1.5
            t_local, sol, phase_vec = self.simulate_segment(current_state, segment, s_star, dt, approx_time, idx)

            times.append(accumulated_time + t_local)
            states.append(sol)
            phases.append(phase_vec)

            accumulated_time += t_local[-1]
            if idx < len(self.segments):
                next_segment = self.segments[idx]
                next_start = next_segment.position(0.0)
                current_state = np.array([next_start[0], next_start[1], 0.0, 0.0, 0.0])
            else:
                last_state = sol[-1].copy()
                last_state[:2] = segment.position(1.0)
                last_state[2:4] = np.zeros(2)
                current_state = np.hstack([last_state[:4], 0.0])

        all_states = np.vstack(states)
        all_times = np.concatenate(times)
        all_phases = np.concatenate(phases)
        return all_times, all_states, all_phases

def plot_corrected_trajectories(controller, time_hist, sol, phases, s_star):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    x = sol[:, 0].copy()
    y = sol[:, 1].copy()
    vx = sol[:, 2]
    vy = sol[:, 3]

    theta = np.linspace(0, 2*np.pi, 400)
    phase_changes = np.where(np.diff(phases) != 0)[0]
    for idx in phase_changes:
        x[idx:idx+2] = np.nan
        y[idx:idx+2] = np.nan

    ax1.plot(x, y, 'b-', linewidth=2, label='Фактическая траектория')
    ax1.plot(2*np.cos(theta), 2*np.sin(theta), 'r--', label='φ₁: окружность')
    ax1.plot(2 + 3*np.cos(theta), 1 + 2*np.sin(theta), 'g--', label='φ₂: эллипс')
    x_parab = np.linspace(-3, 3, 300)
    ax1.plot(x_parab, 0.5*x_parab**2 - 2, 'm--', label='φ₃: парабола')
    ax1.plot(x[0], y[0], 'ko', label='Старт')
    ax1.plot(x[-1], y[-1], 'ro', label='Финиш')
    ax1.set_title(f'Траектория (s* = {s_star})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(time_hist, vx, label='Vx')
    ax2.plot(time_hist, vy, label='Vy')
    ax2.set_title('Скорости')
    ax2.set_xlabel('Время')
    ax2.set_ylabel('Скорость')
    ax2.grid(True)
    ax2.legend()

    phi_errors = []
    for idx, func in enumerate(controller.phi_funcs):
        mask = phases == (idx + 1)
        values = np.full_like(time_hist, np.nan)
        values[mask] = func(x[mask], y[mask])
        phi_errors.append(values)
        ax3.plot(time_hist[mask], values[mask], label=f'φ{idx+1}')

    ax3.set_title('Ошибки по активным участкам')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('φ')
    ax3.grid(True)
    ax3.legend()

    ux = np.gradient(vx, time_hist, edge_order=2)
    uy = np.gradient(vy, time_hist, edge_order=2)
    ax4.plot(time_hist, ux, label='Ux')
    ax4.plot(time_hist, uy, label='Uy')
    ax4.set_title('Управляющие воздействия (оценка)')
    ax4.set_xlabel('Время')
    ax4.set_ylabel('У')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    return fig, phi_errors

def main():
    controller = CorrectedCoordinatedControl2D()
    x0 = np.array([2.0, 0.0, 0.0, 0.0])
    t_grid = np.linspace(0, 40, 4000)
    speeds = [1.0, 3.0, 5.0]

    for s_star in speeds:
        print(f"Симуляция параметризированного 2D алгоритма для s* = {s_star}")
        time_hist, sol, phases = controller.simulate_trajectory(x0, t_grid, s_star)
        fig, phi_errors = plot_corrected_trajectories(controller, time_hist, sol, phases, s_star)

        max_errors = [
            np.nanmax(np.abs(err)) if np.any(~np.isnan(err)) else 0.0
            for err in phi_errors
        ]
        print(f"  Максимальные ошибки: φ₁={max_errors[0]:.4f}, φ₂={max_errors[1]:.4f}, φ₃={max_errors[2]:.4f}")

        output_dir = "/home/leonidas/projects/itmo/Planning-trajectories-of-movement/lab3/images/task1"
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"corrected_2d_s{s_star}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Результаты сохранены для s* = {s_star}")

if __name__ == "__main__":
    main()
