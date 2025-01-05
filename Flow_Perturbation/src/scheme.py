import torch
class Euler():
    def step(self, func, t, dt, y):
        out = y + dt * func(t, y)
        return out

class RK2():
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        out = y + k2
        return out

class RK4():
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k1)
        k3 = dt * func(t + dt / 2.0, y + 1.0 / 2.0 * k2)
        k4 = dt * func(t + dt, y + k3)
        out = y + 1.0 / 6.0 * k1 + 1.0 / 3.0 * k2 + 1.0 / 3.0 * k3 + 1.0 / 6.0 * k4
        return out
        

class Heun():
    def step(self, func, t, dt, y):
        k1 = dt * func(t, y)
        k2 = dt * func(t + dt, y + k1)
        out = y + 0.5 * (k1 + k2)
        return out

class Euler_dSt():
    def step(self, func, t, dt, y, eps):
        k, div_k = func(t, y, eps)
        out = y + dt * k
        div_out = div_k*dt
        return out, div_out

class RK2_dSt():
    def step(self, func, t, dt, y, eps):
        k1, div_k1 = func(t, y, eps)
        k2, div_k2 = func(t + dt / 2.0, y + 1.0 / 2.0 * dt * k1, eps)
        out = y + dt * k2
        div_out = div_k2 * dt
        return out, div_out
    
class RK4_dSt():
    def step(self, func, t, dt, y, eps):
        k1, div_k1 = func(t, y, eps)
        k2, div_k2 = func(t + dt / 2.0, y + 1.0 / 2.0 * dt * k1, eps)
        k3, div_k3 = func(t + dt / 2.0, y + 1.0 / 2.0 * dt * k2, eps)
        k4, div_k4 = func(t + dt, y + dt * k3, eps)
        out = y + 1.0 / 6.0 * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        div_out = 1.0 / 6.0 * (div_k1 + 2 * div_k2 + 2 * div_k3 + div_k4)*dt
        return out, div_out

class Heun_dSt():
    def step(self, func, t, dt, y, eps):
        k1, div_k1 = func(t, y, eps)
        k2, div_k2 = func(t + dt, y + dt * k1, eps)
        out = y + 0.5 * dt * (k1 + k2)
        div_out = 0.5 * (div_k1 + div_k2) * dt
        return out, div_out
    


