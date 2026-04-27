# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold

from mlp_core import mlp_m_3w

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')

x_oryginalne = x_norm.copy()
liczba_cech = x_oryginalne.shape[0]

K1_fixed = 8
K2_fixed = 5
max_epoch = 60
err_goal = 0.05 
disp_freq = 10 
lr_fixed = 1e-4
mc_fixed = 0.8

CVN = 10
skfold = StratifiedKFold(n_splits=CVN, shuffle=True, random_state=42)

print("Ważność głosowań.")

mlp_bazowy = mlp_m_3w(x_oryginalne, y_t, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch, True)
wynik_bazowy = mlp_bazowy.train_CV(CVN, skfold)
print(f"Wynik bazowy (PK): {wynik_bazowy:.2f}%\n")

spadki_pk = []

for i in range(liczba_cech):
    x_zepsute = x_oryginalne.copy()
    
    x_zepsute[i, :] = 0 
    
    mlp_test = mlp_m_3w(x_zepsute, y_t, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch, True)
    wynik_zepsuty = mlp_test.train_CV(CVN, skfold)
    
    spadek = wynik_bazowy - wynik_zepsuty
    spadki_pk.append(spadek)

print("Zakończono.")

plt.figure(figsize=(12, 6))
numery_glosowan = np.arange(1, liczba_cech + 1)

kolory = ['crimson' if s > 0 else 'lightblue' for s in spadki_pk]

plt.bar(numery_glosowan, spadki_pk, color=kolory, edgecolor='black')
plt.axhline(0, color='black', linewidth=1) 
plt.xlabel('Kwestia głosowania', fontsize=12)
plt.ylabel('Spadek skuteczności sieci [%]', fontsize=12)
plt.ylim(min(spadki_pk) - 1, max(spadki_pk) + 1)
# Krótkie opisy głosowań na osi X (możesz edytować listę poniżej)
vote_labels = [
    'handicapped-infants', 'water-project-cost-sharing', 'adoption-budget-res',
    'physician-fee-freeze', 'el-salvador-aid', 'religious-groups-in-schools',
    'anti-satellite-test-ban', 'aid-to-contras', 'mx-missile', 'immigration',
    'synfuels-cutback', 'education-spending', 'superfund', 'crime',
    'duty-free-exports', 'export-admin-act-SA'
]
plt.xticks(numery_glosowan, vote_labels, rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

for i, spadek in enumerate(spadki_pk):
    wysokosc_tekstu = spadek + 0.1 if spadek >= 0 else spadek - 0.4
    plt.text(i + 1, wysokosc_tekstu, f"{spadek:.1f}%", ha='center', va='bottom', fontsize=9)

plt.savefig("Fig.3_Waznosc_Glosowan_kongres.png", bbox_inches='tight')
plt.show()