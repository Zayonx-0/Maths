import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 300)
y1=np.cos(x)
y2=np.sin(x)
plt.plot(x, y1, label='cos(x)')
plt.plot(x, y2, label='sin(x)')
plt.show()