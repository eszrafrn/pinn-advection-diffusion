from initial_conditions import gaussian_pulase
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1,100)
c0 = gaussian_pulase(x, A=1.0, x0=0.5, sigma=0.1)
plt.plot(x, c0)
plt.xlabel('x')
plt.ylabel('c')
plt.title('Kondisi Awal: Pulsa Gaussian')
plt.show()
plt.savefig('Test IC Gaussian_Pulse.png')
