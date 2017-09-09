# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:00:25 2017

@author: calvin-pc
"""

import numpy as np

#figure 2.11 in Evolving intelligent system
class eclustering_plus:
    def __init__(self):
        # Time
        self.t = 0
        self.z_star = []
        self.D = []
        self.b = []
        self.c = []
        self.width = []
        self.A = []
        self.S = []
        self.U = []
        self.L = 1
        self.w = []
        self.rules = 0
        
    # membership of x in rule i
    # Eq 2.2
    def membership(self,i,x):
        c = self.c[i]
        width = self.width[i]
        
        z = np.square(x-c) / (np.square(width)*2)
        return np.exp(- np.sum(z))
    
    # Eq 2.3
    def activation_degree (self,i,x):
        return np.prod(self.membership(i,x))
        
    def transform(self,x):
        inp = np.array(x)
        z = inp
        self.t = self.t+1
        if (self.t == 1):
            self.z_star = [z]
            self.last_rule_D = [1]
            self.b = 0
            self.c = np.zeros(z.shape)
            self.width = [np.ones(z.shape) * 0.5]
            self.A = [1]
            self.S = [1]
            self.U = [1]
            self.L = 1
            self.w = [1]
            self.last_z = z
            self.rules = 1
            return z[:len(x)]
        else:
            # Calculate Eq 2.6
            self.b = self.b + np.sum(np.square(self.last_z))
            self.c = self.c + self.last_z
            divisor = (self.t - 1)*(np.sum(np.square(z)) + 1) + self.b - 2*(np.sum(z*self.c))
            Dt = (self.t - 1)/divisor
            
            # Calculate Eq 2.7
            for i in range(self.rules):
                last_rule_D = self.last_rule_D[i]
                t = self.t
                divisor = t - 1 + (t - 2)*(1/last_rule_D - 1) + np.sum(z - self.last_z)
                self.last_rule_D[i] = (t-1)/divisor
            
            # Using Condition A2
            if (Dt > np.max(self.last_rule_D) or Dt < np.min(self.last_rule_D)):
                # Form new rule
                self.rules = self.rules + 1
                self.z_star.append(z)
                self.last_rule_D.append(1)
                # Check Condition B
                for 
            else:
                # Get the closest rule 
                
            self.last_z = z    