import numpy as np
import hickle as hkl

data = np.genfromtxt('transformed_votes.csv', delimiter=',')

# Oddzielenie etykiet i cech
y_t = data[:, 0].reshape(1, -1)
x = data[:, 1:].T

x_norm = x

# Zapis do formatu HKL
hkl.dump([x, y_t, x_norm, x_norm, y_t], 'kongres.hkl')
print("Zapisano dane do pliku kongres.hkl")