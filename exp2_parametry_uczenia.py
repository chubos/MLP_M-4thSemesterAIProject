# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from tqdm import tqdm

from mlp_core import mlp_m_3w # IMPORT NASZEJ SIECI

# 1. Wczytywanie danych
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
x = x_norm

# 2. Stałe parametry do eksperymentu (Architektura z Exp 1)
K1_fixed = 5
K2_fixed = 8
max_epoch = 40
err_goal = 0.05 
disp_freq = 10 

# 3. Zmienne (siatka lr i mc)
lr_vec = np.logspace(-5, -2, 10) 
mc_vec = np.arange(0.05, 1.0, 0.05)

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# --- EKSPERYMENT 2 ---
start = timer()
print("Rozpoczynam Eksperyment 2: Optymalizacja lr i mc...")
PK_2D_lrmc = np.zeros([len(lr_vec), len(mc_vec)])

for lr_ind in tqdm(range(len(lr_vec)), desc="Postęp lr"):
    for mc_ind in range(len(mc_vec)):
        mlpnet = mlp_m_3w(x, y_t, K1_fixed, K2_fixed,  \
                           lr_vec[lr_ind], err_goal, disp_freq, mc_vec[mc_ind], \
                           max_epoch, True)
        PK_2D_lrmc[lr_ind, mc_ind] = mlpnet.train_CV(CVN, skfold)

print(f"Zakończono w {timer()-start:.2f} s.")

# Najlepszy wynik z całej siatki (lr, mc)
best_idx = np.unravel_index(np.argmax(PK_2D_lrmc), PK_2D_lrmc.shape)
best_lr = lr_vec[best_idx[0]]
best_mc = mc_vec[best_idx[1]]
best_pk = PK_2D_lrmc[best_idx]

print("Najlepsze parametry uczenia:")
print(f"lr = {best_lr:.6g}, mc = {best_mc:.2f}, PK = {best_pk:.2f}%")

# Rysowanie wykresu 3D
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(lr_vec, mc_vec)
surf = ax.plot_surface(np.log10(X), Y, PK_2D_lrmc.T, cmap='viridis')
ax.set_xlabel('log10(lr)')
ax.set_ylabel('mc')
ax.set_zlabel('PK [%]')
ax.view_init(30, 200)
plt.savefig("Fig.2_PK_lrmc_kongres.png", bbox_inches='tight')
plt.show()