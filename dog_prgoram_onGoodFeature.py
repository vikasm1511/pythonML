import numpy as np
import matplotlib.pyplot as plt

greyhound = 500
labs = 500

grey_height = 28 + 4* np.random.randn(greyhound)
labs_height = 24 + 4 * np.random.randn( labs )

plt.hist([grey_height, labs_height], stacked = False, color=['r','b'])
plt.show()
