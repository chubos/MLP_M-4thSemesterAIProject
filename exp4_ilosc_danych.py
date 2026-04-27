# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

from mlp_core import mlp_m_3w # IMPORT NASZEJ SIECI

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')

X_all = x_norm.T  
Y_all = y_t.T     

# 2. Parametry sieci
K1_fixed, K2_fixed = 8, 5
max_epoch = 80
err_goal = 0.05 
disp_freq = 10 
lr_fixed = 1e-4
mc_fixed = 0.80


X_train_pula, X_test, Y_train_pula, Y_test = train_test_split(
    X_all, Y_all, test_size=0.2, stratify=Y_all, random_state=42
)

frakcje = [0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

wyniki_testowe = []
liczba_probek_oX = []


for frac in frakcje:
    if frac < 1.0:
        X_train_sub, _, Y_train_sub, _ = train_test_split(
            X_train_pula, Y_train_pula, train_size=frac, stratify=Y_train_pula, random_state=42
        )
    else:
        X_train_sub, Y_train_sub = X_train_pula, Y_train_pula
        
    aktualna_liczba = X_train_sub.shape[0]
    liczba_probek_oX.append(aktualna_liczba)

    mlp_test = mlp_m_3w(X_train_sub.T, Y_train_sub.T, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch, True)
    mlp_test.train(X_train_sub.T, Y_train_sub.T)

    pred_test = mlp_test.predict(X_test.T)
    pk_test = np.mean(np.where(pred_test >= 0, 1, -1) == Y_test.T) * 100
    wyniki_testowe.append(pk_test)

print("Zakończono.\n")

plt.figure(figsize=(10, 6))
plt.plot(liczba_probek_oX, wyniki_testowe, 'o-', color='mediumseagreen', label='Dane testowe (nieznane)')

plt.xlabel('Liczba kongresmenów w zbiorze uczącym', fontsize=12)
plt.ylabel('Skuteczność PK [%]', fontsize=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim([min(wyniki_testowe)-5, 105]) 

plt.savefig("Fig.4_Ilosc_Danych_kongres.png", bbox_inches='tight')
plt.show()