# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import nnet_jit4 as net
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold, train_test_split

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
                        
            self.d3 = net.deltatan(self.y3, self.e) 
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
            
            self.w1, self.b1 = net.nwtan(self.K1, self.L)
            self.w2, self.b2 = net.nwtan(self.K2, self.K1)
            self.w3, self.b3 = net.nwtan(self.K3, self.K2)
            self.w1_t_1, self.b1_t_1 = self.w1.copy(), self.b1.copy()
            self.w2_t_1, self.b2_t_1 = self.w2.copy(), self.b2.copy()
            self.w3_t_1, self.b3_t_1 = self.w3.copy(), self.b3.copy()
            self.SSE = 0
            self.SSE_vec = []
            self.PK_vec = []
            
            self.train(x_train.T, y_train.T)
            result = self.predict(x_test.T)
            n_test_samples = test.size
            PK_vec[i] = np.mean(np.where(result >= 0, 1, -1) == y_test) * 100
            
        return np.mean(PK_vec)


if __name__ == '__main__':
    print("Test sieci bazowej")
    x, y_t, x_norm, x_n_s, y_t_s = hkl.load('kongres.hkl')
    K1_test, K2_test = 4, 4
    lr_test, mc_test = 1e-3, 0.9
    err_goal_test = 0.05
    disp_freq_test = 10
    max_epoch_test = 400
    initialize_test = True
    
    test_net = mlp_m_3w(x_norm, y_t, K1_test, K2_test, lr_test, err_goal_test, disp_freq_test, mc_test, max_epoch_test, initialize_test)

    X = x_norm.T
    y = np.squeeze(y_t)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    x_train = X_train.T
    x_test = X_test.T
    y_train = y_train.reshape(1, -1)
    y_test = y_test.reshape(1, -1)

    test_net.train(x_train, y_train)
    wyniki = test_net.predict(x_test)
    decyzje = np.where(wyniki >= 0, 1, -1)
    pk_test = np.mean(decyzje == y_test) * 100
    print(f"\nSkuteczność na zbiorze testowym (20%): {pk_test:.2f}%")

    # Wykres testowy krzywej uczenia
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(test_net.SSE_vec) + 1), test_net.SSE_vec, color='red')
    plt.grid(True)
    plt.savefig('sse_test_core.png', dpi=300, bbox_inches='tight')
    plt.close()


