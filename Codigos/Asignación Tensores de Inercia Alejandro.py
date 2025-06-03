import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv("datosmasas.csv")
df = df.rename(columns={'masas': 'mass'})

# Masa total
M = df['mass'].sum()
print("Masa total:", M)

# Centro de masa 2D
x_cm = (df['mass'] * df['x']).sum() / M
y_cm = (df['mass'] * df['y']).sum() / M
print("Centro de masa 2D:", (x_cm, y_cm))

# Coordenadas relativas
x_rel = df['x'] - x_cm
y_rel = df['y'] - y_cm

# Tensor de inercia 2D
I_xx = (df['mass'] * y_rel**2).sum()
I_yy = (df['mass'] * x_rel**2).sum()
I_xy = - (df['mass'] * x_rel * y_rel).sum()
I_tensor_2d = np.array([[I_xx, I_xy],
                        [I_xy, I_yy]])
print("Tensor de inercia 2D:\n", I_tensor_2d)

# Autovalores y autovectores 2D
eigvals_2d, eigvecs_2d = np.linalg.eig(I_tensor_2d)
print("Autovalores 2D:", eigvals_2d)
print("Autovectores 2D:\n", eigvecs_2d)

# Centro de masa 3D
z_cm = (df['mass'] * df['z']).sum() / M
print("Centro de masa 3D:", (x_cm, y_cm, z_cm))

# Coordenadas relativas 3D
x_rel3 = df['x'] - x_cm
y_rel3 = df['y'] - y_cm
z_rel3 = df['z'] - z_cm

# Tensor de inercia 3D
I_tensor_3d = np.zeros((3,3))
for i in range(len(df)):
    m = df.loc[i, 'mass']
    r = np.array([x_rel3[i], y_rel3[i], z_rel3[i]])
    I_tensor_3d += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

print("Tensor de inercia 3D:\n", I_tensor_3d)

# Autovalores y autovectores 3D
eigvals_3d, eigvecs_3d = np.linalg.eig(I_tensor_3d)
print("Autovalores 3D:", eigvals_3d)
print("Autovectores 3D:\n", eigvecs_3d)
