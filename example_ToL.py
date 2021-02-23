import numpy as np
import matplotlib.pyplot as plt

Points = 200+1
Lambda = np.linspace(400, 800, Points)

def example(lambda0=600, sigma=75):
    saveycol = [",".join(wavelength) for wavelength in [Lambda.astype(str)]][0]
    spectrum_1 = (1 - 0.8 * np.exp(-np.square((Lambda - lambda0)/sigma)/2))
    sigma /= 2
    spectrum_2 = (1 - 0.8 * np.exp(-np.square((Lambda - lambda0) / sigma) / 2))
    sigma /= 2
    spectrum_3 = (1 - 0.8 * np.exp(-np.square((Lambda - lambda0) / sigma) / 2))
    spec = np.array([spectrum_1,spectrum_2,spectrum_3])

    np.savetxt("sigma.csv", spec, header=saveycol, fmt='%f', delimiter=',')

    #plt.plot(Lambda, spectrum)
    #plt.show()

example()