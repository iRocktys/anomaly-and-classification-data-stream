import numpy as np
from scipy.optimize import minimize
from collections import deque

class DSPOT:
    def __init__(self, q=1e-3, depth=50, t_quantile=0.98): 
        self.q = q
        self.d = depth
        self.t_quantile = t_quantile 
        self.W = deque(maxlen=depth)
        self.peaks = []
        self.t = None
        self.zq = None
        self.k = 0
        self.N_t = 0
        self.gamma = None
        self.sigma = None
        self.init_buffer = []
        self.M = 0.0

    def _log_likelihood(self, gamma, sigma, Y):
        if sigma <= 0:
            return -np.inf
        if gamma == 0:
            return -len(Y) * np.log(sigma) - np.sum(Y) / sigma
        
        val = 1 + (gamma / sigma) * Y
        if np.any(val <= 0):
            return -np.inf
        return -len(Y) * np.log(sigma) - (1 + 1/gamma) * np.sum(np.log(val))

    def _grimshaw(self, Y):
        Y = np.array(Y)
        Y_m, Y_M, Y_mean = np.min(Y), np.max(Y), np.mean(Y)

        def u(x):
            return np.mean(1 / (1 + x * Y))
        def v(x):
            return 1 + np.mean(np.log1p(x * Y))
        def w(x):
            return (u(x) * v(x) - 1)**2

        epsilon = 1e-8
        bounds1 = (-1/Y_M + epsilon, -epsilon)
        
        if Y_m < epsilon:
            Y_m = epsilon
            
        min_bound2 = 2 * (Y_mean - Y_m) / (Y_mean * Y_m)
        max_bound2 = 2 * (Y_mean - Y_m) / (Y_m**2)
        bounds2 = (min_bound2, max_bound2 + epsilon) 

        candidates = [0.0]
        
        for bnds in [bounds1, bounds2]:
            if bnds[0] >= bnds[1]: continue
            start_points = np.linspace(bnds[0], bnds[1], 3)
            for x0 in start_points:
                res = minimize(w, x0, bounds=[bnds], method='L-BFGS-B')
                if res.success:
                    candidates.append(res.x[0])
        
        best_ll = -np.inf
        best_gamma, best_sigma = 0.1, np.std(Y) if np.std(Y) > 0 else 0.1
        
        for x in candidates:
            if x == 0:
                gamma, sigma = 0.0, Y_mean
            else:
                gamma = v(x) - 1
                sigma = gamma / x if x != 0 else Y_mean
            ll = self._log_likelihood(gamma, sigma, Y)
            if ll > best_ll:
                best_ll = ll
                best_gamma = gamma
                best_sigma = sigma
                
        return best_gamma, best_sigma

    def _calc_threshold(self):
        if self.gamma == 0:
            return self.t - self.sigma * np.log(self.q * self.k / self.N_t)
        else:
            return self.t + (self.sigma / self.gamma) * (((self.q * self.k / self.N_t)**(-self.gamma)) - 1)

    def calibrate(self, warmup_data):
        for i in range(min(self.d, len(warmup_data))):
            self.W.append(warmup_data[i])
            
        X_prime = []
        for i in range(self.d, len(warmup_data)):
            M_i = np.mean(self.W)
            x_p = warmup_data[i] - M_i
            X_prime.append(x_p)
            self.W.append(warmup_data[i])
            
        X_prime = np.array(X_prime)
        if len(X_prime) == 0:
            X_prime = np.array(warmup_data)
            
        # 2. Utilizando o parâmetro t_quantile em vez do valor fixo 0.98
        self.t = np.quantile(X_prime, self.t_quantile) 
        
        self.peaks = [xp - self.t for xp in X_prime if xp > self.t]
        self.N_t = len(self.peaks)
        self.k = len(X_prime)
        
        if self.N_t > 0:
            self.gamma, self.sigma = self._grimshaw(self.peaks)
            self.zq = self._calc_threshold()
        else:
            self.zq = self.t + 0.1
            self.gamma, self.sigma = 0.1, 0.1
            
        self.M = np.mean(self.W)

    def update_and_predict(self, score, warmup_instances=0):
        if len(self.init_buffer) < warmup_instances:
            self.init_buffer.append(score)
            if len(self.init_buffer) == warmup_instances:
                self.calibrate(self.init_buffer)
            return 0, 0.0, 0.0
        
        X_i = score
        X_prime_i = X_i - self.M
        is_anomaly = 0
        
        if X_prime_i > self.zq:
            is_anomaly = 1
        elif X_prime_i > self.t:
            self.peaks.append(X_prime_i - self.t)
            self.N_t += 1
            self.k += 1
            self.gamma, self.sigma = self._grimshaw(self.peaks)
            self.zq = self._calc_threshold()
            self.W.append(X_i)
            self.M = np.mean(self.W)
        else:
            self.k += 1
            self.W.append(X_i)
            self.M = np.mean(self.W)
            
        return is_anomaly, self.M + self.zq, self.M