#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
    
    def control(self, setpoint, current_value, dt):
        error = setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MassSpringDamper:
    def __init__(self, m, k, c):
        self.m = m
        self.k = k
        self.c = c
    
    def update(self, force, dt):
        a = force / self.m
        return a

# Parámetros del sistema
m = 1.0  # Masa (kg)
k = 10.0  # Constante del resorte (N/m)
c = 1.0   # Coeficiente de amortiguación (Ns/m)

# Crear objetos de controlador PID y sistema
controller = PIDController(Kp=20, Ki=5, Kd=10)
system = MassSpringDamper(m, k, c)

# Parámetros de simulación
dt = 0.01  # Paso de tiempo
total_time = 10.0  # Tiempo total de simulación
num_steps = int(total_time / dt)

# Condiciones iniciales
current_position = 0.0
current_velocity = 0.0

# Listas para almacenar datos de la simulación
time_values = np.arange(0, total_time, dt)
position_values = []

# Simulación
for _ in range(num_steps):
    # Calcular la fuerza de control utilizando el controlador PID
    target_position = 1.0  # Posición deseada
    control_force = controller.control(target_position, current_position, dt)
    
    # Actualizar la posición y la velocidad del sistema utilizando la dinámica del sistema
    acceleration = system.update(control_force, dt)
    current_velocity += acceleration * dt
    current_position += current_velocity * dt
    
    # Almacenar la posición actual para su visualización
    position_values.append(current_position)

# Visualizar resultados
plt.plot(time_values, position_values, label='Posición')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.title('Control PID de un sistema masa-resorte-amortiguador')
plt.legend()
plt.grid(True)
plt.show()
