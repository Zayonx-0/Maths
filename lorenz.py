import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Définition du système de Lorenz
def lorenz_system(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Paramètres du système de Lorenz
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Conditions initiales
initial_state = [1.0, 1.0, 1.0]

# Intervalle de temps et points de calcul
t_span = (0, 50)  # Temps total
t_eval = np.linspace(t_span[0], t_span[1], 10000)  # Points de temps

# Résolution numérique avec solve_ivp
solution = solve_ivp(
    lorenz_system,
    t_span,
    initial_state,
    args=(sigma, beta, rho),
    t_eval=t_eval,
    method='RK45'
)

# Extraction des résultats
x, y, z = solution.y

# Configuration de la visualisation interactive
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(min(x), max(x))
ax.set_ylim(min(y), max(y))
ax.set_zlim(min(z), max(z))
ax.set_title("Attracteur de Lorenz (Temps réel)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Initialisation de la ligne qui se forme
line, = ax.plot([], [], [], lw=0.7, color="blue")

# Fonction de mise à jour pour l'animation
def update(num):
    line.set_data(x[:num], y[:num])  # Met à jour X et Y
    line.set_3d_properties(z[:num])  # Met à jour Z
    return line,

# Création de l'animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    interval=1,  # Intervalle entre les frames en millisecondes
    blit=False
)

plt.show()
