# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 19:39:05 2016

@author: calvin-pc
"""

import numpy as np

def Sigmoid(a,b,x):
    # sig' for sigmoidal function, G(a,b,x) = 1/(1+exp(-(ax+b)))
    return np.sum(np.array(a)*x) + b


"Online sequential extreme learning machine with forgetting mechanism"
class FOSELM:    
    def __init__(self,p,s,L,n,G = Sigmoid):
        # Jumlah OS-ELM
        self.p = p
        # Timeliness
        self.s = s
        # Jumlah neuron dalam satu OS-ELM
        self.L = L
        # Panjang Input
        self.n = n
        # Fungsi aktivasi
        self.G = G
        # Parameter OS-ELM, shape(pXLXn)
        self.a = np.random.uniform(size=(p,L,n))
        # Parameter OS-ELM, shape(pXL)
        self.b = np.random.uniform(size=(p,L))
        
        self.P = [None for i in range(p)]
        self.beta = [None for i in range(p)]
        self.HTH = [[] for i in range(p)]
        self.HTT = [[] for i in range(p)]
        self.sumHTH = [None for i in range(p)]
        self.sumHTT = [None for i in range(p)]
        
    def train(self,x,t):
        self.trainMany([x],[t])
                
    def trainMany(self,xs,ts):
        for r in range(0,self.p):
            H = np.array([self.calculateHiddenNeuron(r,x) for x in xs])
            T = np.array(ts)
            HTH = np.dot(H.T,H)
            HTT = np.dot(H.T,T)
            self.HTH[r].append(HTH)
            self.HTT[r].append(HTT)
            if (len(self.HTH[r]) >= self.s):
                if (len(self.HTH[r]) == self.s):
                    self.sumHTH[r] = sum(self.HTH[r])
                    self.sumHTT[r] = sum(self.HTT[r])
                else:
                    first_HTH = self.HTH[r][0]
                    first_HTT = self.HTT[r][0]
                    self.sumHTH[r] = self.sumHTH[r] + HTH - first_HTH
                    self.sumHTT[r] = self.sumHTT[r] + HTT - first_HTT
                #self.P[r] = np.linalg.pinv(sum(np.dot(H.T,H) for H in self.H[r]))
                #self.beta[r] = np.dot(self.P[r],sum((np.dot(H.T,t) for H,t in zip(self.H[r],self.T))))
                #self.P[r] = np.linalg.pinv(sum(self.HTH[r]))
                #self.beta[r] = np.dot(self.P[r],sum(self.HTT[r]))
                self.P[r] = np.linalg.pinv(self.sumHTH[r])
                self.beta[r] = np.dot(self.P[r],self.sumHTT[r])
                self.HTT[r] = self.HTT[r][-self.s:]
                self.HTH[r] = self.HTH[r][-self.s:]
            
    def output(self,x):
        return sum(self.calculateOutput(r,x) for r in range(self.p))/self.p
        
    # Calculate result of hidden neuron of j-th OS-ELM
    # return [G(a1,b1,x) .. G(aL,bL,x)]
    def calculateHiddenNeuron(self,j,x):
        return [self.G(A,B,x) for A,B in zip(self.a[j],self.b[j])]
        
    def calculateOutput(self,j,x):
        HL = self.calculateHiddenNeuron(j,x)
        ret = sum((beta*g for beta,g in zip(self.beta[j],HL)))
        return ret
        
class EFOSELM:    
    def __init__(self,p,f,L,n,G = Sigmoid):
        # Jumlah OS-ELM
        self.p = p
        self.f = f
        # Jumlah neuron dalam satu OS-ELM
        self.L = L
        # Panjang Input
        self.n = n
        # Fungsi aktivasi
        self.G = G
        # Parameter OS-ELM, shape(pXLXn)
        self.a = np.random.uniform(size=(p,L,n))
        # Parameter OS-ELM, shape(pXL)
        self.b = np.random.uniform(size=(p,L))
        
        self.P = [None for i in range(p)]
        self.beta = [None for i in range(p)]
        self.sumHTH = [None for i in range(p)]
        self.sumHTT = [None for i in range(p)]
        
    def train(self,x,t):
        self.trainMany([x],[t])
                
    def trainMany(self,xs,ts):
        for r in range(0,self.p):
            H = np.array([self.calculateHiddenNeuron(r,x) for x in xs])
            T = np.array(ts)
            HTH = np.dot(H.T,H)
            HTT = np.dot(H.T,T)
            if (self.sumHTH[r] == None):
                self.sumHTH[r] = HTH
                self.sumHTT[r] = HTT
            else:
                self.sumHTH[r] = self.f*self.sumHTH[r] + HTH
                self.sumHTT[r] = self.f*self.sumHTT[r] + HTT
            #self.P[r] = np.linalg.pinv(sum(np.dot(H.T,H) for H in self.H[r]))
            #self.beta[r] = np.dot(self.P[r],sum((np.dot(H.T,t) for H,t in zip(self.H[r],self.T))))
            #self.P[r] = np.linalg.pinv(sum(self.HTH[r]))
            #self.beta[r] = np.dot(self.P[r],sum(self.HTT[r]))
            self.P[r] = np.linalg.pinv(self.sumHTH[r])
            self.beta[r] = np.dot(self.P[r],self.sumHTT[r])
            
    def output(self,x):
        return sum(self.calculateOutput(r,x) for r in range(self.p))/self.p
        
    # Calculate result of hidden neuron of j-th OS-ELM
    # return [G(a1,b1,x) .. G(aL,bL,x)]
    def calculateHiddenNeuron(self,j,x):
        return [self.G(A,B,x) for A,B in zip(self.a[j],self.b[j])]
        
    def calculateOutput(self,j,x):
        HL = self.calculateHiddenNeuron(j,x)
        ret = sum((beta*g for beta,g in zip(self.beta[j],HL)))
        return ret
        
class EOSELM:    
    def __init__(self,p,L,n,G = Sigmoid):
        # Jumlah OS-ELM
        self.p = p
        # Jumlah neuron dalam satu OS-ELM
        self.L = L
        # Panjang Input
        self.n = n
        # Fungsi aktivasi
        self.G = G
        # Parameter OS-ELM, shape(pXLXn)
        self.a = np.random.uniform(size=(p,L,n))
        # Parameter OS-ELM, shape(pXL)
        self.b = np.random.uniform(size=(p,L))
        
        self.P = [None for i in range(p)]
        self.beta = [None for i in range(p)]
        
    def train(self,x,t):
        self.trainMany([x],[t])
                
    def trainMany(self,xs,ts):
        for r in range(0,self.p):
            H = np.array([self.calculateHiddenNeuron(r,x) for x in xs])
            T = np.array(ts)
            
            if (self.P[r] == None):
                self.P[r] = np.linalg.pinv(np.dot(H.T,H))
                self.beta[r] = np.dot(np.dot(self.P[r],H.T),T)
            else:
                P = np.matrix(self.P[r])
                H = np.matrix(H)
                beta = np.matrix(self.beta[r])
                T = np.matrix(T)
                next_P = P - P * H.T * (np.identity(H.shape[0]) + H*P*H.T).I * H * P
                A = T- (H*beta)
                next_beta = beta + next_P * H.T * A
                self.P[r] = np.array(next_P)
                self.beta[r] = np.array(next_beta)
            
    def output(self,x):
        return sum(self.calculateOutput(r,x) for r in range(self.p))/self.p
        
    # Calculate result of hidden neuron of j-th OS-ELM
    # return [G(a1,b1,x) .. G(aL,bL,x)]
    def calculateHiddenNeuron(self,j,x):
        return [self.G(A,B,x) for A,B in zip(self.a[j],self.b[j])]
        
    def calculateOutput(self,j,x):
        HL = self.calculateHiddenNeuron(j,x)
        ret = sum((beta*g for beta,g in zip(self.beta[j],HL)))
        return ret