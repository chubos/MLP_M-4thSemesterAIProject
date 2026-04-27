# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold

from mlp_core import mlp_m_3w 

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
x = x_norm

max_epoch = 400
err_goal = 0.05 
disp_freq = 10 
lr_fixed = 1e-3
mc_fixed = 0.9

K1_vec = np.arange(1, 11, 1)
K2_vec = K1_vec

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

print("Optymalizacja K1 i K2.")
PK_2D_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])

for k1_ind in range(len(K1_vec)):
    for k2_ind in range(len(K2_vec)):
        mlpnet = mlp_m_3w(x, y_t, K1_vec[k1_ind], K2_vec[k2_ind],  \
                           lr_fixed, err_goal, disp_freq, mc_fixed, \
                           max_epoch, True)
        PK_2D_K1K2[k1_ind, k2_ind] = mlpnet.train_CV(CVN, skfold)

print("Zakończono.")

best_idx = np.unravel_index(np.argmax(PK_2D_K1K2), PK_2D_K1K2.shape)
best_k1 = K1_vec[best_idx[0]]
best_k2 = K2_vec[best_idx[1]]
best_pk = PK_2D_K1K2[best_idx]

print("Najlepsze ustawienie architektury:")
print(f"K1 = {best_k1}, K2 = {best_k2}, PK = {best_pk:.2f}%")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')
ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('PK [%]')
ax.set_zlim(90, 100)
ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_kongres.png", bbox_inches='tight')
plt.show()