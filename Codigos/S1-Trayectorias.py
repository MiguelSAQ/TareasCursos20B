import numpy as np
import matplotlib.pyplot as plt

# Parámetros
B = 1       # Campo magnético
Ey = 1      # Campo eléctrico
c = 1       # Velocidad de la luz
A_values = [2, 0.5, 1]  # Diferentes valores de A para los tres casos
omega_c = 1 # Frecuencia ciclotrón
omega_L = omega_c # Considerando el caso especial donde omega_L = omega_c
t = np.linspace(0, 20, 1000)  # Tiempo

titles = [r'Trayectoria para $A > |cE_y/B|$', r'Trayectoria para $A < |cE_y/B|$', r'Trayectoria para $A = |cE_y/B|$']
colors = ['r', 'g', 'b']

for i, A in enumerate(A_values):
    x = (A / omega_c) * np.sin(omega_L * t) + (c * Ey / B) * t
    y = (A / omega_c) * (np.cos(omega_L * t) - 1)
    
    plt.figure(figsize=(6, 5))
    plt.plot(x, y, color=colors[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(titles[i])
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid()
    plt.show()