import numpy as np
from numpy import dot

class KalmanFilterClass(object):
    def __init__(self):
        self.dt = 5e-3
        self.A = np.array([[1,0],[0,1]])
        self.u = np.zeros((2,1))
        self.b = np.array([[0],[255]])
        self.P = np.diag((3.0, 3.0))
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.Q = np.eye(self.u.shape[0])
        self.R = np.eye(self.b.shape[0])
        self.lastResult = np.array([[0], [255]])
       
    def predict(self):
        self.u = np.round(dot(self.F, self.u))
        self.P = dot(self.F, dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u
        return self.u

    def correct(self, b, flag):
        if not flag:
            self.b = self.lastResult
        else:
            self.b = b
        C = dot(self.A, dot(self.P, self.A.T))+self.R
        K = dot(self.P, dot(self.A.T, np.linalg.inv(C)))
        self.u = np.round(self.u + dot(K, (self.b - dot(self.A, self.u))))
        self.P = self.P - dot(K, dot(C, K.T))
        self.lastResult = self.u
        return self.u
