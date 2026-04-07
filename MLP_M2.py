# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import nnet_jit4 as net
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from tqdm import tqdm  # Dodane do paska postępu
        
class mlp_m_3w:
    def __init__(self, x, y_t, K1, K2, lr, err_goal, \
                 disp_freq, mc, max_epoch, initialize):
        self.x          = x
        self.L          = self.x.shape[0] 
        self.y_t        = y_t
        self.K1         = K1
        self.K2         = K2
        self.lr         = lr
        self.err_goal   = err_goal
        self.disp_freq  = disp_freq
        self.mc         = mc
        self.max_epoch  = max_epoch
        self.K3         = y_t.shape[0] if len(y_t.shape)>1 else 1
        self.SSE_vec    = [] 
        self.PK_vec     = []
        self.data       = self.x.T
        self.target     = self.y_t
        self.initialize = initialize

        if self.initialize: 
            self.w1, self.b1 = net.nwtan(self.K1, self.L)  
            self.w2, self.b2 = net.nwtan(self.K2, self.K1)
            self.w3, self.b3 = net.nwtan(self.K3, self.K2)
            hkl.dump([self.w1,self.b1,self.w2,self.b2,self.w3,self.b3], 'wagi3w.hkl')
        else:
            self.w1,self.b1,self.w2,self.b2,self.w3,self.b3 = hkl.load('wagi3w.hkl')
            
        self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3
        self.SSE = 0
        self.lr_vec = list()
    
    def predict(self,x):
        n = np.dot(self.w1, x)
        self.y1 = net.tansig( n,  self.b1*np.ones(n.shape)) 
        n = np.dot(self.w2, self.y1)
        self.y2 = net.tansig( n,  self.b2*np.ones(n.shape))
        n = np.dot(self.w3, self.y2)
        self.y3 = net.tansig(n, self.b3*np.ones(n.shape)) 
        return self.y3
        
    def train(self, x_train, y_train):
        for epoch in range(1, self.max_epoch+1): 
            self.y3 = self.predict(x_train)    
            self.e = y_train - self.y3 
        
            self.SSE_t_1 = self.SSE
            self.SSE = net.sumsqr(self.e) 
            self.PK = np.mean(np.where(self.y3 >= 0, 1, -1) == y_train) * 100
            self.PK_vec.append(self.PK)
            if self.SSE < self.err_goal or self.PK == 100: 
                break 
            
            if np.isnan(self.SSE): 
                break
                        
            self.d3 = net.deltatan_out(self.y3, self.e) 
            self.d2 = net.deltatan(self.y2, self.d3, self.w3)
            self.d1 = net.deltatan(self.y1, self.d2, self.w2) 
            self.dw1, self.db1 = net.learnbp(x_train,  self.d1, self.lr) 
            self.dw2, self.db2 = net.learnbp(self.y1,  self.d2, self.lr)
            self.dw3, self.db3 = net.learnbp(self.y2,  self.d3, self.lr)
            
            self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp = \
            self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy(), self.w3.copy(), self.b3.copy()
            
            self.w1 += self.dw1 + self.mc * (self.w1 - self.w1_t_1)
            self.b1 += self.db1 + self.mc * (self.b1 - self.b1_t_1) 
            self.w2 += self.dw2 + self.mc * (self.w2 - self.w2_t_1) 
            self.b2 += self.db2 + self.mc * (self.b2 - self.b2_t_1) 
            self.w3 += self.dw3 + self.mc * (self.w3 - self.w3_t_1) 
            self.b3 += self.db3 + self.mc * (self.b3 - self.b3_t_1) 
            
            self.w1_t_1, self.b1_t_1, self.w2_t_1, self.b2_t_1, self.w3_t_1, self.b3_t_1 = \
            self.w1_temp, self.b1_temp, self.w2_temp, self.b2_temp, self.w3_temp, self.b3_temp
            
            self.SSE_vec.append(self.SSE) 
            
    def train_CV(self, CV, skfold):
        PK_vec = np.zeros(CV)
        
        for i, (train, test) in enumerate(skfold.split(self.data, np.squeeze(self.target)), start=0):
            x_train, x_test = self.data[train], self.data[test]
            y_train, y_test = np.squeeze(self.target)[train], np.squeeze(self.target)[test]
            
            self.train(x_train.T, y_train.T)
            result = self.predict(x_test.T)
        
            n_test_samples = test.size
            PK_vec[i] = np.mean(np.where(result >= 0, 1, -1) == y_test) * 100
            
        PK = np.mean(PK_vec)
        return PK

# ==========================================
# GŁÓWNA CZĘŚĆ SKRYPTU
# ==========================================

# Wczytywanie danych z pliku HKL
x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
x = x_norm

max_epoch = 200
err_goal = 0.25 
disp_freq = 10 

# Zagęszczona siatka hiperparametrów
lr_vec = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
mc_vec = np.arange(0.05, 1.0, 0.04) # Od 0.05 co 0.04
K1_vec = np.arange(1, 11, 1)        # Neurony od 1 do 10
K2_vec = K1_vec

n_runs = 1 # <--- LICZBA POWTÓRZEŃ (1 = pojedyncze przejście)

start = timer()
CVN = 10
skfold = StratifiedKFold(n_splits=CVN)

# --- EKSPERYMENT 1: Szukanie K1 i K2 ---
print(f"Rozpoczynam Eksperyment 1: Optymalizacja K1 i K2 ({n_runs} przejście)...")
PK_2D_K1K2 = np.zeros([len(K1_vec), len(K2_vec)])
PK_2D_K1K2_max = 0
k1_ind_max = 0 
k2_ind_max = 0

for k1_ind in tqdm(range(len(K1_vec)), desc="Postęp K1"):
    for k2_ind in range(len(K2_vec)):
        pk_temp = [] # Lista do przechowywania wyników z n_runs prób
        for _ in range(n_runs):
            mlpnet = mlp_m_3w(x, y_t, K1_vec[k1_ind], K2_vec[k2_ind],  \
                               lr_vec[-1], err_goal, disp_freq, mc_vec[-1], \
                                   max_epoch, True)
            PK = mlpnet.train_CV(CVN, skfold)
            pk_temp.append(PK)
            
        PK_mean = np.mean(pk_temp) # Dla n_runs=1 jest to wynik z pojedynczego przejścia
        PK_2D_K1K2[k1_ind, k2_ind] = PK_mean
        if PK_mean > PK_2D_K1K2_max:
            PK_2D_K1K2_max = PK_mean
            k1_ind_max = k1_ind 
            k2_ind_max = k2_ind

# Rysowanie wykresu 3D dla K1 i K2
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(K1_vec, K2_vec)
surf = ax.plot_surface(X, Y, PK_2D_K1K2.T, cmap='viridis')
ax.set_xlabel('K1')
ax.set_ylabel('K2')
ax.set_zlabel('Średnie PK [%]')
ax.view_init(30, 200)
plt.savefig("Fig.1_PK_K1K2_kongres.png", bbox_inches='tight')

# --- EKSPERYMENT 2: Szukanie lr i mc ---
print(f"\nRozpoczynam Eksperyment 2: Optymalizacja lr i mc ({n_runs} przejście)...")
PK_2D_lrmc = np.zeros([len(lr_vec), len(mc_vec)])
PK_2D_lrmc_max = 0
lr_ind_max = 0 
mc_ind_max = 0

for lr_ind in tqdm(range(len(lr_vec)), desc="Postęp lr"):
    for mc_ind in range(len(mc_vec)):
        pk_temp = [] # Lista do przechowywania wyników z n_runs prób
        for _ in range(n_runs):
            mlpnet = mlp_m_3w(x, y_t, K1_vec[k1_ind_max], K2_vec[k2_ind_max],  \
                               lr_vec[lr_ind], err_goal, disp_freq, mc_vec[mc_ind], \
                                   max_epoch, True)
            PK = mlpnet.train_CV(CVN, skfold)
            pk_temp.append(PK)
            
        PK_mean = np.mean(pk_temp) # Dla n_runs=1 jest to wynik z pojedynczego przejścia
        PK_2D_lrmc[lr_ind, mc_ind] = PK_mean
        if PK_mean > PK_2D_lrmc_max:
            PK_2D_lrmc_max = PK_mean
            lr_ind_max = lr_ind 
            mc_ind_max = mc_ind

# Rysowanie wykresu 3D dla lr i mc
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(lr_vec, mc_vec)
surf = ax.plot_surface(np.log10(X), Y, PK_2D_lrmc.T, cmap='viridis')
ax.set_xlabel('log10(lr)')
ax.set_ylabel('mc')
ax.set_zlabel('Średnie PK [%]')
ax.view_init(30, 200)
plt.savefig("Fig.2_PK_lrmc_kongres.png", bbox_inches='tight')
            
print("\nOPTYMALNE WARTOŚCI: K1={} | K2={} | lr={} | mc={:.2f} | PK={:.2f}%".\
      format(K1_vec[k1_ind_max], K2_vec[k2_ind_max], lr_vec[lr_ind_max], \
             mc_vec[mc_ind_max], PK_2D_lrmc[lr_ind_max, mc_ind_max]))            

print("Czas wykonywania eksperymentów:", timer()-start, "sekund")

# ==========================================
# NOWA CZĘŚĆ: RYSOWANIE KRZYWYCH UCZENIA 
# ==========================================
print("\nTrenowanie ostatecznego modelu w celu narysowania krzywych uczenia...")

best_net = mlp_m_3w(x, y_t, K1_vec[k1_ind_max], K2_vec[k2_ind_max],  \
                    lr_vec[lr_ind_max], err_goal, disp_freq, mc_vec[mc_ind_max], \
                    max_epoch, True)

# Trenujemy na całości, aby zebrać historię uczenia do wykresów
best_net.train(x, y_t)

# Wykres 3: Krzywa błędu (SSE)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(best_net.SSE_vec) + 1), best_net.SSE_vec, color='red', linewidth=2)
plt.title('Krzywa uczenia - Błąd średniokwadratowy (SSE) w czasie')
plt.xlabel('Epoka (Iteracja)')
plt.ylabel('Błąd SSE')
plt.grid(True)
plt.savefig("Fig.3_Krzywa_SSE_kongres.png", bbox_inches='tight')

# Wykres 4: Krzywa poprawności klasyfikacji (PK)
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(best_net.PK_vec) + 1), best_net.PK_vec, color='green', linewidth=2)
plt.title('Krzywa uczenia - Skuteczność klasyfikacji (PK) w czasie')
plt.xlabel('Epoka (Iteracja)')
plt.ylabel('Skuteczność PK [%]')
plt.grid(True)
plt.savefig("Fig.4_Krzywa_PK_kongres.png", bbox_inches='tight')

# Wyświetlamy wszystkie wykresy na końcu
plt.show()

# ==========================================
# TESTOWANIE OSTATECZNEGO MODELU (Zwracanie -1 lub 1)
# ==========================================
print("\n--- TESTOWANIE NAJLEPSZEGO MODELU ---")

# Odpytujemy naszą najlepszą sieć o pierwsze 5 próbek (kongresmenów)
# Zmienna 'x' to nasze dane wejściowe
surowe_wyniki = best_net.predict(x[:, :5]) 

# Magia dzieje się tutaj: zamieniamy wyniki na twarde -1 lub 1
ostateczne_decyzje = np.where(surowe_wyniki >= 0, 1, -1)

# Prawdziwe etykiety dla porównania
prawdziwe_partie = y_t[:, :5].astype(int)

print("Surowe wyniki z sieci (wartości ciągłe):")
print(np.round(surowe_wyniki, 3))

print("\nOstateczne decyzje (twarde -1 lub 1):")
print(ostateczne_decyzje)

print("\nPrawdziwe partie w danych:")
print(prawdziwe_partie)