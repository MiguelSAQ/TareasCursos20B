import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de la ecuación
C = 10      # Puedes modificar este valor
a = 0.261799 # Ángulo en radianes
theta_0 = np.pi #Ángulo inicial

# Definir un rango de valores para r
r = np.linspace(11, 20, 300)  # r debe ser mayor que C para evitar problemas con arcsin

# Calcular theta(r) y z(r)
theta = -1 / np.sin(a) * np.arcsin(C / np.abs(r)) + theta_0
z = (1 / np.tan(a)) * r  # Usamos la identidad cot(x) = 1/tan(x)

# Convertir a coordenadas cartesianas
x = r * np.cos(theta)
y = r * np.sin(theta)

# Graficar la trayectoria en 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, label='Trayectoria de la partícula', color='b')

# Configurar la escala uniforme
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Etiquetas de ejes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Movimiento de la Partícula en Coordenadas Cilíndricas")

ax.legend()
plt.show()