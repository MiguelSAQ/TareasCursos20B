import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parámetros físicos
rho = 1000      # Densidad del líquido (kg/m³)
g = 9.81        # Aceleración gravitatoria (m/s²)
C = 0.47        # Coeficiente de arrastre (depende de la forma de la burbuja)
P0 = 101325     # Presión atmosférica (Pa)
V0 = 1e-6       # Volumen inicial de la burbuja (m³)
h0 = -5          # Profundidad inicial (m)
m = 1e-6        # Masa efectiva de la burbuja (kg)

# Definimos la ecuación diferencial
def eq_dif(t, y):
    h, v = y
    # Ecuación para la aceleración (d^2h/dt^2 = dv/dt)
    term1 = (rho * g * V0 / m) * (P0 + rho * g * h0) / (P0 + rho * g * h) 
    term2 = (C * rho * np.pi / (2 * m)) * ((3/(4 * np.pi)) * V0 * (P0 + rho * g * h0) / (P0 + rho * g * h))**(2/3) * v**2
    dhdt = v
    dvdt = term1 - term2 - g
    return [dhdt, dvdt]

# Condiciones iniciales
h_init = h0
v_init = 0  # Burbuja comienza en reposo

# Tiempo para la simulación
t_span = (0, 30)  # Simularemos por 30 segundos
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Tiempo de evaluación para la simulación

# Resolver la ecuación diferencial con el método de Runge-Kutta (solve_ivp)
sol = solve_ivp(eq_dif, t_span, [h_init, v_init], t_eval=t_eval, method='RK45')

# Extraer los resultados
h = sol.y[0]
v = sol.y[1]
t = sol.t

# Graficar la velocidad de la burbuja y la posición en función del tiempo
fig, ax1 = plt.subplots()

ax1.set_xlabel('Tiempo (s)')
ax1.set_ylabel('Velocidad (m/s)', color='tab:red')
ax1.plot(t, v, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Posición (m)', color='tab:blue')
ax2.plot(t, h, color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

# Línea gris punteada en h = 0
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Línea vertical punteada en t = 9.057
ax1.axvline(x=9.057, color='gray', linestyle='--', linewidth=1)

plt.title('Velocidad y Posición de la burbuja en función del tiempo')
plt.show()