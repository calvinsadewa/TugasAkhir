# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:31:26 2017

@author: calvin-pc
"""
import numpy as np

class InputStandardizer:
    def __init__(self,machine,cap_n = 999999):
        self.mean_x = 0
        self.std_x = 0
        self.mean_y = 0
        self.std_y = 0
        self.t = 0
        self.machine = machine
        self.cap_n = cap_n
        
    def train(self,x,d):
        x = np.array(x)
        d = np.array(d)
        self.updateStandarizer(x,d)
        X = self.scaleX(x)
        Y = self.scaleY(d)
        self.machine.train(X,Y)
    def output(self,x):
        x = np.array(x)
        Y = self.machine.output(self.scaleX(x))
        return self.rescaleY(Y)
        
    def updateStandarizer(self,x,d):
        y = d
        self.t = self.t+1
        t = self.t
        if (t == 1):
            self.mean_x = x
            self.std_x = np.zeros(x.shape) + 0.0000000000001
            self.mean_y = d
            self.std_y = np.zeros(d.shape) + 0.0000000000001
        else:
            if (self.cap_n < t):
                t = self.cap_n
                self.t = self.cap_n
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
            mean_x = self.mean_x
            mean_y = self.mean_y
            std_x = self.std_x
            std_y = self.std_y
            self.mean_x = mean_x + (x - mean_x) / t
            self.mean_y = mean_y + (y - mean_y) / t
            self.std_x = np.sqrt((t-1)*np.square(std_x)/t + (x-mean_x)*(x-mean_x)/(t-1))
            self.std_y = np.sqrt((t-1)*np.square(std_y)/t + (y-mean_y)*(y-mean_y)/(t-1))
    
    def scale(self,x,mean_x,std_x):
        return (x-mean_x)/(std_x)
    
    def scaleX(self,x):
        return self.scale(x,self.mean_x,self.std_x)
        
    def scaleY(self,y):
        return self.scale(y,self.mean_y,self.std_y)
        
    def rescaleY(self,y):
        return y*self.std_y +self.mean_y