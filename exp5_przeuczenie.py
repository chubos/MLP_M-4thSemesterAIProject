# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from mlp_core import mlp_m_3w # IMPORT NASZEJ SIECI

# 1. Wczytywanie danych
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
X_all, Y_all = x_norm.T, y_t.T     

# Aby wywołać przeuczenie: dajemy bardzo mało danych do nauki i dużo do testów
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all, test_size=0.8, stratify=Y_all, random_state=42
)

# Wracamy do macierzy [cechy, próbki]
x_tr, y_tr = X_train.T, Y_train.T
x_te, y_te = X_test.T, Y_test.T

# 2. Parametry sprzyjające przeuczeniu (duża pojemność modelu)
K1_fixed, K2_fixed = 12, 12
lr_fixed = 1e-3
mc_fixed = 0.9

# KLUCZ: Uczymy tylko 1 epokę na iterację pętli, err_goal bardzo niski, żeby nie przerwało samo
err_goal = 0.00001 
max_epoch_single = 1 
disp_freq = 10 
liczba_epok_total = 180 # Dłuższa obserwacja ułatwia uchwycenie przeuczenia

print("--- EKSPERYMENT 5: ZJAWISKO PRZEUCZENIA (OVERFITTING) ---")
print(f"Rozpoczynam badanie krok po kroku przez {liczba_epok_total} epok...")

mlp_net = mlp_m_3w(x_tr, y_tr, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch_single, True)

historia_train_sse = []
historia_test_sse = []

for epoka in range(liczba_epok_total):
    # Wykonuje dokładnie jedną epokę uczenia
    mlp_net.train(x_tr, y_tr)
    
    # Zapisujemy błąd na zbiorze treningowym
    historia_train_sse.append(mlp_net.SSE)
    
    # Ręcznie oceniamy błąd na ukrytym zbiorze testowym
    pred_test = mlp_net.predict(x_te)
    blad_test = y_te - pred_test
    sse_test = np.sum(blad_test**2) # Równowartość net.sumsqr
    historia_test_sse.append(sse_test)

print("Zakończono.\n")

historia_train_sse = np.array(historia_train_sse)
historia_test_sse = np.array(historia_test_sse)

# 3. Rysowanie wykresu Overfittingu
plt.figure(figsize=(10, 6))

epoki = np.arange(1, liczba_epok_total + 1)
plt.plot(epoki, historia_train_sse, color='blue', linewidth=2, label='Błąd Treningowy (SSE)')
plt.plot(epoki, historia_test_sse, color='red', linewidth=2, label='Błąd Testowy (SSE)')

plt.title('Zjawisko Przeuczenia (Overfitting) - Błąd w czasie', fontsize=14)
plt.xlabel('Epoka uczenia', fontsize=12)
plt.ylabel('Suma Kwadratów Błędów (SSE)', fontsize=12)

# Zaznaczenie momentu, w którym należy przerwać naukę (najniższy błąd testowy)
optymalna_epoka = int(np.argmin(historia_test_sse) + 1)
najmniejszy_blad = float(np.min(historia_test_sse))
plt.axvline(optymalna_epoka, color='black', linestyle=':', label=f'Early Stopping (Epoka {optymalna_epoka})')
plt.plot(optymalna_epoka, najmniejszy_blad, 'ko') # Czarna kropka w optimum

# Podświetlamy obszar po minimum błędu testowego jako strefę przeuczenia
plt.axvspan(optymalna_epoka, liczba_epok_total, color='salmon', alpha=0.15, label='Strefa potencjalnego przeuczenia')

if optymalna_epoka < liczba_epok_total:
    wzrost_po_min = historia_test_sse[-1] - najmniejszy_blad
    plt.text(
        optymalna_epoka + 2,
        najmniejszy_blad,
        f'Wzrost SSE test po minimum: {wzrost_po_min:.2f}',
        fontsize=9,
        color='darkred'
    )

plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig("Fig.5_Przeuczenie_kongres.png", bbox_inches='tight')
plt.show()