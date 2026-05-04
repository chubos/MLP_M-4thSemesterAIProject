# -*- coding: utf-8 -*-
"""
Skrypt do generowania wykresów z zapisanych wyników GridSearch.
Pozwala na modyfikację opisów i ponowne generowanie wykresów bez uruchamiania całego GridSearch.

Uruchomienie:
    python exp5_plot_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Wczytanie wyników z pliku
results_file = 'exp5_gridsearch_results.json'

if not os.path.exists(results_file):
    print(f"Błąd: Plik '{results_file}' nie istnieje!")
    print("Uruchom najpierw exp5_gridsearch_4d.py")
    exit(1)

print(f"Wczytywanie wyników z pliku '{results_file}'...")
with open(results_file, 'r') as f:
    results_data = json.load(f)

results = results_data['results']
K1_vec = np.array(results_data['K1_vec'])
K2_vec = np.array(results_data['K2_vec'])
lr_vec = np.array(results_data['lr_vec'])
mc_vec = np.array(results_data['mc_vec'])

print(f"Wczytano {len(results)} wyników.")

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

# Generowanie wykresów
print("\nGenerowanie wykresów...")

# 1. PK vs (K1, K2) - uśredniając po lr i mc
PK_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])
for i, k1 in enumerate(K1_vec):
    for j, k2 in enumerate(K2_vec):
        subset = [r['PK'] for r in results if r['K1'] == k1 and r['K2'] == k2]
        PK_K1K2[i, j] = np.mean(subset) if subset else 0

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(PK_K1K2, cmap='viridis', aspect='auto')
ax.set_xlabel('K2', fontsize=12)
ax.set_ylabel('K1', fontsize=12)
ax.set_xticks(range(len(K2_vec)))
ax.set_yticks(range(len(K1_vec)))
ax.set_xticklabels(K2_vec)
ax.set_yticklabels(K1_vec)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('PK [%]', fontsize=11)
plt.tight_layout()
plt.savefig('Fig.5a_GridSearch_K1K2.png', dpi=150, bbox_inches='tight')
print("✓ Fig.5a zapisany")
plt.close()

# 2. PK vs (lr, mc) - uśredniając po K1 i K2
PK_lrmc = np.zeros([len(lr_vec), len(mc_vec)])
for i, lr in enumerate(lr_vec):
    for j, mc in enumerate(mc_vec):
        subset = [r['PK'] for r in results if abs(r['lr'] - lr) < 1e-10 and abs(r['mc'] - mc) < 1e-10]
        PK_lrmc[i, j] = np.mean(subset) if subset else 0

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(PK_lrmc, cmap='viridis', aspect='auto')
ax.set_xlabel('mc', fontsize=12)
ax.set_ylabel('log10(lr)', fontsize=12)
ax.set_xticks(range(0, len(mc_vec), 2))
ax.set_xticklabels([f"{mc:.2f}" for mc in mc_vec[::2]])
ax.set_yticks(range(0, len(lr_vec), 2))
ax.set_yticklabels([f"{np.log10(lr):.1f}" for lr in lr_vec[::2]])
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('PK [%]', fontsize=11)
plt.tight_layout()
plt.savefig('Fig.5b_GridSearch_lrmc.png', dpi=150, bbox_inches='tight')
print("✓ Fig.5b zapisany")
plt.close()

# 3. Histogram rozkładu PK
fig, ax = plt.subplots(figsize=(8, 6))
pk_values = [r['PK'] for r in results]
ax.hist(pk_values, bins=30, color='steelblue', edgecolor='black')
ax.set_xlabel('PK [%]', fontsize=12)
ax.set_ylabel('Liczba kombinacji', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fig.5c_GridSearch_histogram.png', dpi=150, bbox_inches='tight')
print("✓ Fig.5c zapisany")
plt.close()

# 4. PK vs K1 (wszystkie kombinacje)
fig, ax = plt.subplots(figsize=(8, 6))
for k1 in K1_vec:
    pk_for_k1 = [r['PK'] for r in results if r['K1'] == k1]
    ax.scatter([k1] * len(pk_for_k1), pk_for_k1, alpha=0.4, s=30, color='steelblue')
ax.set_xlabel('K1', fontsize=12)
ax.set_ylabel('PK [%]', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fig.5d_GridSearch_K1_scatter.png', dpi=150, bbox_inches='tight')
print("✓ Fig.5d zapisany")
plt.close()

# 5. PK vs K2 (wszystkie kombinacje)
fig, ax = plt.subplots(figsize=(8, 6))
for k2 in K2_vec:
    pk_for_k2 = [r['PK'] for r in results if r['K2'] == k2]
    ax.scatter([k2] * len(pk_for_k2), pk_for_k2, alpha=0.4, s=30, color='steelblue')
ax.set_xlabel('K2', fontsize=12)
ax.set_ylabel('PK [%]', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fig.5e_GridSearch_K2_scatter.png', dpi=150, bbox_inches='tight')
print("✓ Fig.5e zapisany")
plt.close()


print("\n✓ Wszystkie wykresy zostały wygenerowane pomyślnie!")
print("\nMożesz teraz modyfikować opisy w TEX-ie bez konieczności powtarzania GridSearch.")
