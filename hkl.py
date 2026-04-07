import numpy as np
import hickle as hkl

# 1. Wczytanie danych z pliku CSV 
# Zakładam, że plik nazywa się 'transformed_votes.csv' i nie ma nagłówków
data = np.genfromtxt('transformed_votes.csv', delimiter=',')

# 2. Oddzielenie etykiet i cech
y_t = data[:, 0].reshape(1, -1)
x = data[:, 1:].T

# --- TUTAJ JAWNIE DEFINIUJEMY x_norm ---
# Ponieważ dane to -1, 0, 1, normalizacja nie jest potrzebna, 
# więc x_norm to po prostu kopia x.
x_norm = x

# 3. Zapis do formatu HKL
hkl.dump([x, y_t, x_norm, x_norm, y_t], 'kongres.hkl')
print("Gotowe! Zapisano dane do pliku kongres.hkl")