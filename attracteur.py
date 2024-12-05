import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Définition du système chaotique
def system(t, state):
    x, y, z = state
    dxdt = -y - z
    dydt = x + 0.2 * y
    dzdt = 0.2 + z * (x - 5.7)
    return [dxdt, dydt, dzdt]

# Conditions initiales
initial_state = [-0.5, 0.1, 0.7]
t_span = (0, 200)  # Temps
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Points de temps

# Résolution numérique
solution = solve_ivp(system, t_span, initial_state, t_eval=t_eval, method='RK45')

# Extraction des résultats
x, y, z = solution.y

# Visualisation 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, color='red', linewidth=0.7)
ax.set_title("Attracteur chaotique")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
