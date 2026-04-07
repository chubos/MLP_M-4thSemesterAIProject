# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from tqdm import tqdm

from mlp_core import mlp_m_3w # IMPORT NASZEJ SIECI BAZOWEJ

# 1. Wczytywanie danych
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
X_all, Y_all = x_norm.T, y_t.T     

# 2. Aby wywołać przeuczenie: dajemy mało danych do nauki (30%) i dużo do testów (70%)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all, test_size=0.7, stratify=Y_all, random_state=42
)

# Wracamy do macierzy [cechy, próbki]
x_tr, y_tr = X_train.T, Y_train.T
x_te, y_te = X_test.T, Y_test.T

# --- SZTUCZNE WPROWADZENIE SZUMU ---
# Odwracamy 20% etykiet w zbiorze treningowym, żeby zmusić sieć do wkuwania "bzdur" na pamięć
procent_szumu = 0.40
liczba_do_zmiany = int(x_tr.shape[1] * procent_szumu)

np.random.seed(42) # Stałe ziarno, żeby zawsze psuło tych samych
indeksy_szumu = np.random.choice(x_tr.shape[1], liczba_do_zmiany, replace=False)

# Jeśli było 0 to robi się 1, jeśli 1 to robi się 0
y_tr[0, indeksy_szumu] = 1 - y_tr[0, indeksy_szumu]
print(f"Celowo zepsuto etykiety dla {liczba_do_zmiany} kongresmenów ze zbioru treningowego.")
# -----------------------------------

# 3. Parametry sprzyjające przeuczeniu (zbyt duża sieć dla tak małego zbioru)
K1_fixed, K2_fixed = 8, 8
lr_fixed = 1e-3
mc_fixed = 0.9

# KLUCZ: Uczymy tylko 1 epokę na iterację pętli, err_goal bardzo niski, żeby nie przerwało samo
err_goal = 0.00001 
max_epoch_single = 1 
disp_freq = 10 
liczba_epok_total = 150 # Ile kroków zrobimy łącznie

print("--- EKSPERYMENT 6: PRZEUCZENIE PRZY ZASZUMIONYCH ETYKIETACH ---")
print(f"Rozpoczynam badanie krok po kroku przez {liczba_epok_total} epok...")

mlp_net = mlp_m_3w(x_tr, y_tr, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch_single, True)

historia_train_sse = []
historia_test_sse = []

start = timer()

for epoka in tqdm(range(liczba_epok_total), desc="Trenowanie"):
    # Wykonuje dokładnie jedną epokę uczenia
    mlp_net.train(x_tr, y_tr)
    
    # Zapisujemy błąd na zbiorze treningowym (wliczając nasz szum)
    historia_train_sse.append(mlp_net.SSE)
    
    # Ręcznie oceniamy błąd na CZYSTYM, ukrytym zbiorze testowym
    pred_test = mlp_net.predict(x_te)
    blad_test = y_te - pred_test
    sse_test = np.sum(blad_test**2) # Równowartość net.sumsqr
    historia_test_sse.append(sse_test)

print(f"Zakończono w {timer()-start:.2f} s.\n")

# 4. Rysowanie wykresu Overfittingu
plt.figure(figsize=(10, 6))

plt.plot(range(1, liczba_epok_total + 1), historia_train_sse, color='blue', linewidth=2, label='Błąd Treningowy (SSE)')
plt.plot(range(1, liczba_epok_total + 1), historia_test_sse, color='red', linewidth=2, label='Błąd Testowy (SSE)')

plt.title('Zjawisko Przeuczenia (Overfitting) - Błąd w czasie', fontsize=14)
plt.xlabel('Epoka uczenia', fontsize=12)
plt.ylabel('Suma Kwadratów Błędów (SSE)', fontsize=12)

# Zaznaczenie momentu, w którym należy przerwać naukę (najniższy błąd testowy)
optymalna_epoka = np.argmin(historia_test_sse) + 1
najmniejszy_blad = min(historia_test_sse)

plt.axvline(optymalna_epoka, color='black', linestyle=':', label=f'Early Stopping (Epoka {optymalna_epoka})')
plt.plot(optymalna_epoka, najmniejszy_blad, 'ko') # Czarna kropka w optimum

plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("Fig.6_Przeuczenie_kongres_szum.png", bbox_inches='tight')
plt.show()