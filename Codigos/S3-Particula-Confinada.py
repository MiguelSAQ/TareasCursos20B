import numpy as np
import matplotlib.pyplot as plt

# Parámetros del potencial
V0 = 1.0    # Profundidad del pozo de potencial
alpha = 1.0 # Parámetro del potencial
x = np.linspace(-3, 3, 500)  # Rango de x

# Función del potencial
V_x = -V0 / np.cosh(alpha * x)**2

# Niveles de energía
E_values = [-0.8 * V0, -0.5 * V0, 0, 0.2 * V0]
colors = ['blue', 'green', 'red', 'purple']
labels = [r'$E = -0.8V_0$', r'$E = -0.5V_0$', r'$E = 0$ (umbral de escape)', r'$E = 0.2V_0$']

# Crear la figura
plt.figure(figsize=(8, 6))

# Graficar las curvas de energía cinética T(x) = E - V(x)
for E, color, label in zip(E_values, colors, labels):
    T_x = E - V_x  # Energía cinética
    plt.plot(x, T_x, color=color, linestyle='-', linewidth=2, label=label)

# Línea roja en T = 0 (donde la energía cinética se anula)
plt.axhline(0, color='red', linestyle='dotted', linewidth=2, label=r'$T(x) = 0$ (puntos de retorno)')

# Etiquetas y formato
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$T(x) = E - V(x)$', fontsize=14)
plt.title('Energía Cinética de la Partícula en el Pozo de Potencial', fontsize=14)
plt.legend()
plt.grid(True)
plt.ylim(-0.2, 1.2)  # Ajuste del eje Y

# Mostrar la gráfica
plt.show()
