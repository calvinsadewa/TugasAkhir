# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:59:05 2016

@author: calvin
"""

import numpy as np
import copy

class SOFNN:
    # initial_width
    initial_width = 0
    # length of input
    r = 0
    # number of neuron
    n = 0
    # Center matrix, shape(r,n)
    cM = None
    # Width (sigma) Matrix, shape(r,n)
    sM = None
    # Q matrix
    Q = None
    # Theta, shape (1,M)
    theta = None
    # Width enchancement factor
    k = None
    # Previous input vector
    xs = np.asmatrix([[]])
    # previous targets
    ds = []
    # krmse , expected training root mean squared error
    krmse = None
    # predifened value 0 < lambd < 1
    lambd = None
    # delta, pruning treshold
    delta = None
    window = None
    n_update = 0
    
    def __init__(self, r,delta = 0.05,krmse = 0.01,initial_width=0.4,k=1.12,lambd=0.8,window=120):
        self.r = r
        self.initial_width = initial_width
        self.delta = delta
        self.k = k
        self.krmse = krmse
        self.lambd = lambd
        self.window = window

    # x : input vector
    # d : target
    # refer to Evolving Intelligent Systems: Methodology and Applications page 214
    # Mutating selfS
    def train(self,x,d):
        assert(len(x) == self.r)
        if self.n == 0:
            self.xs = np.asmatrix([x])
        else:
            self.xs = np.vstack((self.xs[-self.window:],x))
        self.ds = self.ds[-self.window:]
        self.ds.append(d)
        if self.n == 0 :
            # Initialize first neuron
            # Eq 9.14
            self.cM = np.array(x).reshape(self.r,1)
            # Eq 9.15
            self.sM = self.cM.copy()
            self.sM.fill(self.initial_width)
            self.n = 1
            self.updateStructure()
        else:
            # Update Q and theta
            # Eq 9.29 - 9.32
            pT = self.pT(x)
            p = pT.T
            # Eq 9.29
            L = self.Q*p*((1+pT*self.Q*p).I)
            # Eq 9.32
            e = d - pT*self.theta
            theta_t = self.theta + L*e
            epsilon = d - p.T*theta_t
            alpha = int((np.abs(e) >= np.abs(epsilon)))
            # Eq 9.30
            identity = np.identity(self.Q.shape[0])
            self.Q = (identity - alpha*L*pT)*self.Q
            # Eq 9.31
            self.theta = self.theta + alpha * L * e
        stop = False
        while (not stop):
            # Calculate error and if-part criteria
            # Eq 9.10, Eq 9.13
            # Eq 9.10
            err = np.abs(d - self.output(x))
            # Eq 9.13
            phis = self.phi(x)
            max_index = np.argmax(phis)
            if (err <= self.delta and phis[max_index] > 0.1354):
                stop = True
                continue
            
            if (err <= self.delta and phis[max_index] < 0.1354):
                #Enlarge the width, Eq 9.21
                while (self.phi(x)[max_index] < 0.1354):
                    for i in range(0,self.r):
                        self.sM[i,max_index] *= self.k
                self.updateStructure()
                continue
            
            if (err > self.delta):
                if (phis[max_index] < 0.1354):
                    #Enlarge the width, Eq 9.21
                    copy_sm = copy.deepcopy(self.sM)
                    while (self.phi(x)[max_index] < 0.1354):
                        for i in range(0,self.r):
                            self.sM[i,max_index] *= self.k
                    after_err = np.abs(d - self.output(x))
                    if (after_err > self.delta):
                        # useless
                        self.sM = copy_sm
                    else:
                        self.updateStructure()
                        stop = True
                        continue
                # Prune Neurons? Eq 9.33, Eq 9.34
                while(True and self.n > 1):
                    erms = self.erms()
                    target_erms = max(self.lambd*erms,self.krmse)
                    H = np.linalg.pinv(self.Q)
                    delta_thetas = [np.zeros((self.theta.shape)) for i in range(0,self.n)]
                    for i in range(0,self.n):
                        start = i * (self.r+1)
                        for j in range(0,self.r + 1):
                            ind = start + j
                            delta_thetas[i][ind,0] = -self.theta[ind,0]
                    delta_erms = [(float(0.5*dt.T*H*dt),num) for num, dt in enumerate(delta_thetas)]
                    sdelta_erms = sorted(delta_erms)
                    csM,ccM,cQ,ctheta = copy.deepcopy((self.sM,self.cM,self.Q,self.theta))
                    
                    # delete neuron
                    _,num = sdelta_erms[0]

                    self.n -= 1
                    self.cM = np.delete(self.cM,num,axis=1)
                    self.sM = np.delete(self.sM,num,axis=1)
                    self.updateStructure()
                    deleted_erms = self.erms()
                    
                    if (deleted_erms < target_erms):
                        # neuron should be deleted
                        continue
                    else:
                        # restore neuron
                        self.sM = csM
                        self.cM = ccM
                        self.Q = cQ
                        self.theta = ctheta
                        self.n += 1
                        break
                # Add neuron
                # Eq 9.16
                # using this Development of Cognitive Capabilities for Smart Home using a Self-Organizing Fuzzy Neural Network 
                all_dist = np.abs(np.matrix(x).T - self.cM)
                Dist = np.min(all_dist,axis=1)
                
                c_next = np.matrix(x).T
                s_next = 0.0000000000001 + Dist
                self.cM = np.hstack((self.cM,c_next))
                self.sM = np.hstack((self.sM,s_next))
                self.n += 1
                self.updateStructure()
            stop = True
        
    
    # Eq 9.2
    # Output of layer 2
    # return [phi1 .. phiN]
    # where N is number of neuron
    def phi(self,x):
        xt = np.asmatrix(x).T
        
        z = np.square(xt - self.cM)
        z2 = z/(np.square(self.sM)*2)
        ret = np.exp(-np.sum(z2,axis=0).astype(float)).A1
        assert(self.n == len(ret))
        return ret
    
    # Eq 9.3
    # Output of layer 3
    # return [psi1 .. psiN]
    # where N is number of neuron
    def psi(self,x):
        phi = self.phi(x)
        div = np.sum(phi)
        if (div == 0):
            div = 1
        ret = phi / div
        assert(len(ret) == self.n)
        return ret
    
    # Eq 9.9
    # Consist [[psi1,psi1x0,..,psi1xR,..psiN,..psiNxR]]
    # where N is how many neuron
    # where R is input length
    # Shape (1,Nx(R+1))
    def pT(self,x):
        psi = self.psi(x)
        mult = np.concatenate(([1], x))
        ret =  np.asmatrix(np.ndarray.flatten(np.asmatrix(psi).T*mult)).astype(float)
        assert(ret.shape == (1,self.n*(self.r+1)))
        return ret

    # create GreatPhi transpose
    def P(self):
        ret = np.asmatrix([self.pT(x).A1 for x in self.xs.getA()]).astype(float)
        assert(ret.shape == (len(self.xs),self.n*(self.r+1)))
        return ret
        
    def D(self):
        return np.matrix(self.ds).T
    
    def updateStructure(self):
        self.n_update += 1
        # Update Parameter, Q and Theta
        P = self.P()
        # Eq 9.25
        self.Q = np.linalg.pinv(P.T*P)
        # Eq 9.24
        D = self.D()
        self.theta = self.Q*P.T*D
        
    def erms(self):
        D = self.D()
        P = self.P()
        erms = 0.5*np.sum(np.square(D - P * self.theta))
        return erms
        
    # Eq 9.6
    # Prediction of SOFNN, output of layer 4
    def output(self,x):
        # Eq 9.6
        return float(self.pT(x)*self.theta)
        
class MIMOSOFNN():
    machines = []
    def __init__(self, r, rt , delta=0.05, krmse = 0.01, initial_width=0.4,k=1.12,lambd=0.8,window=120):
        self.r = r
        self.initial_width = initial_width
        self.delta = delta
        self.k = k
        self.krmse = krmse
        self.lambd = lambd
        self.window = window
        self.rt = rt
        self.n_train = 0
        self.machines = [SOFNN(r, delta, krmse, initial_width,k,lambd,window) for i in range(0,rt)]
        
    def max_n(self):
        return max([m.n for m in self.machines])
    
    def total_n(self):
        return sum([m.n for m in self.machines])
        
    def train(self,xs,ys):
        self.n_train +=1
        assert (self.rt == len(ys))
        for y,machine in zip(ys,self.machines):
            machine.train(xs,y)
        
    def output(self,xs):
        ret = [machine.output(xs) for machine in self.machines]
        assert (self.rt == len(ret))
        return ret