# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
from sklearn.model_selection import StratifiedKFold
import itertools
import json
import os

from mlp_core import mlp_m_3w

x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
x = x_norm

max_epoch = 400
err_goal = 1 
disp_freq = 10 

K1_vec = np.arange(1, 11, 1)
K2_vec = np.arange(1, 11, 1)
lr_vec = np.logspace(-5, -2, 10)
mc_vec = np.arange(0.05, 1.0, 0.05)

CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# Sprawdzenie czy plik z wynikami już istnieje
results_file = 'exp5_gridsearch_results.json'

if os.path.exists(results_file):
    print(f"Wczytywanie wyników z pliku '{results_file}'...")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    results = results_data['results']
    print(f"Wczytano {len(results)} wyników.")
else:
    print("Optymalizacja K1, K2, lr i mc (GridSearch 4D).")
    print(f"Liczba kombinacji do sprawdzenia: {len(K1_vec) * len(K2_vec) * len(lr_vec) * len(mc_vec)}")

    results = []
    counter = 0
    total = len(K1_vec) * len(K2_vec) * len(lr_vec) * len(mc_vec)

    for k1, k2, lr, mc in itertools.product(K1_vec, K2_vec, lr_vec, mc_vec):
        counter += 1
        if counter % 50 == 0:
            print(f"Postęp: {counter}/{total}")
        
        mlpnet = mlp_m_3w(x, y_t, k1, k2, lr, err_goal, disp_freq, mc, max_epoch, True)
        pk = mlpnet.train_CV(CVN, skfold)
        
        results.append({
            'K1': int(k1),
            'K2': int(k2),
            'lr': float(lr),
            'mc': float(mc),
            'PK': float(pk)
        })

    print("Zakończono GridSearch.")
    
    # Zapisanie wyników do pliku JSON
    results_data = {
        'results': results,
        'K1_vec': [int(x) for x in K1_vec.tolist()],
        'K2_vec': [int(x) for x in K2_vec.tolist()],
        'lr_vec': [float(x) for x in lr_vec.tolist()],
        'mc_vec': [float(x) for x in mc_vec.tolist()]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"Wyniki zapisane do pliku '{results_file}'")
    
    # Konwersja z powrotem do numpy arrays
    K1_vec = np.array(results_data['K1_vec'])
    K2_vec = np.array(results_data['K2_vec'])
    lr_vec = np.array(results_data['lr_vec'])
    mc_vec = np.array(results_data['mc_vec'])

# Sortowanie wyników po skuteczności
results_sorted = sorted(results, key=lambda x: x['PK'], reverse=True)

# Wyświetlenie top 10 wyników
print("\nTop 10 najlepszych kombinacji parametrów:")
print("-" * 70)
for i, res in enumerate(results_sorted[:10], 1):
    print(f"{i}. K1={res['K1']:2d}, K2={res['K2']:2d}, lr={res['lr']:.6g}, mc={res['mc']:.2f} -> PK = {res['PK']:.2f}%")

# Najlepsza kombinacja
best = results_sorted[0]
print("\n" + "="*70)
print("NAJLEPSZA KOMBINACJA:")
print(f"K1 = {best['K1']}, K2 = {best['K2']}, lr = {best['lr']:.6g}, mc = {best['mc']:.2f}")
print(f"PK = {best['PK']:.2f}%")
print("="*70)

