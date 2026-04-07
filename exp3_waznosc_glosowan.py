# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from tqdm import tqdm

from mlp_core import mlp_m_3w # IMPORT NASZEJ SIECI BAZOWEJ

# 1. Wczytywanie danych
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')

# Zabezpieczenie danych oryginalnych przed nadpisaniem
x_oryginalne = x_norm.copy()
liczba_cech = x_oryginalne.shape[0] # Powinno wynosić 16 (ilość głosowań)

# 2. Stałe parametry (możesz wpisać tu najlepsze z Exp 1 i Exp 2)
K1_fixed = 5
K2_fixed = 8
max_epoch = 40
err_goal = 0.05 
disp_freq = 10 
lr_fixed = 1e-3
mc_fixed = 0.9

CVN = 10
# Ustawiamy ziarno losowości (random_state), żeby podział na zbiory był zawsze taki sam
# Ułatwi to uczciwe porównanie wyników przy psuciu poszczególnych cech
skfold = StratifiedKFold(n_splits=CVN, shuffle=True, random_state=42)

print("--- EKSPERYMENT 3: WAŻNOŚĆ GŁOSOWAŃ (Ablation Study) ---")

# KROK 1: Wyliczenie wyniku bazowego (na wszystkich danych)
start = timer()
print("Trenowanie modelu bazowego (wszystkie 16 głosowań)...")
mlp_bazowy = mlp_m_3w(x_oryginalne, y_t, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch, True)
wynik_bazowy = mlp_bazowy.train_CV(CVN, skfold)
print(f"Wynik bazowy (PK): {wynik_bazowy:.2f}%\n")

# KROK 2: Testowanie każdej cechy po kolei
spadki_pk = [] # Lista przechowująca, o ile pogorszył się wynik

print("Rozpoczynam analizę i wymazywanie poszczególnych głosowań...")
for i in tqdm(range(liczba_cech), desc="Sprawdzanie głosowań"):
    # Tworzymy świeżą kopię danych
    x_zepsute = x_oryginalne.copy()
    
    # "Usuwamy" informację o i-tym głosowaniu, wymuszając wartość 0 (brak głosu) u każdego polityka
    x_zepsute[i, :] = 0 
    
    # Trenujemy nową sieć na zepsutych danych
    mlp_test = mlp_m_3w(x_zepsute, y_t, K1_fixed, K2_fixed, lr_fixed, err_goal, disp_freq, mc_fixed, max_epoch, True)
    wynik_zepsuty = mlp_test.train_CV(CVN, skfold)
    
    # Liczymy stratę skuteczności
    spadek = wynik_bazowy - wynik_zepsuty
    spadki_pk.append(spadek)

print(f"\nZakończono w {timer()-start:.2f} sekund.")

# KROK 3: Rysowanie wykresu słupkowego
plt.figure(figsize=(12, 6))
numery_glosowan = np.arange(1, liczba_cech + 1)

# Kolorujemy słupki: czerwone jeśli spadek był ponadprzeciętny, szare jeśli był niewielki
srednia_spadkow = np.mean(spadki_pk)
kolory = ['crimson' if s > srednia_spadkow else 'lightslategray' for s in spadki_pk]

plt.bar(numery_glosowan, spadki_pk, color=kolory, edgecolor='black')
plt.axhline(0, color='black', linewidth=1) 
plt.title('Ważność poszczególnych głosowań dla decyzji sieci neuronowej', fontsize=14)
plt.xlabel('Numer Głosowania (1-16)', fontsize=12)
plt.ylabel('Spadek skuteczności sieci [%]\n(Wyższy słupek = Ważniejsze głosowanie)', fontsize=12)
plt.xticks(numery_glosowan)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dodawanie konkretnych procentów nad słupkami
for i, spadek in enumerate(spadki_pk):
    # Jeśli spadek jest na minusie (czyli usunięcie cechy wręcz pomogło), przesuwamy napis pod słupek
    wysokosc_tekstu = spadek + 0.3 if spadek >= 0 else spadek - 1.0
    plt.text(i + 1, wysokosc_tekstu, f"{spadek:.1f}%", ha='center', va='bottom', fontsize=9)

plt.savefig("Fig.3_Waznosc_Glosowan_kongres.png", bbox_inches='tight')
print("Wykres został zapisany jako 'Fig.3_Waznosc_Glosowan_kongres.png'")
plt.show()