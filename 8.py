import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 30)
y = np.cos(x)

plt.plot(x, y)
plt.xlabel("abscisses")
plt.ylabel("ordonnees")
plt.show()
