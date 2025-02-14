import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir el radio a constante
a = 1

# Crear valores de theta
theta = np.linspace(0, np.pi, 100)

# Parametrización en coordenadas cilíndricas
r = a * np.ones_like(theta)  # r es constante y igual a a
z = theta  # z = theta, como indicamos antes

# Convertir a coordenadas cartesianas
x = r * np.cos(theta)
y = r * np.sin(theta)

# Crear la figura y el gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la geodésica (curva)
ax.plot(x, y, z, label='Geodésica')
ax.scatter([a], [0], [0], color='red', label='P1 (a, 0, 0)')
ax.scatter([-a], [0], [np.pi], color='green', label='P2 (a, π, π)')

# Etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Geodésica sobre el cilindro')

# Configurar la escala uniforme
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0

mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Mostrar la leyenda
ax.legend()

# Mostrar el gráfico
plt.show()