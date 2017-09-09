# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sympy import *
from sympy.stats import *
from collections import namedtuple

Kondisi = namedtuple("Kondisi", "Nama Pr Total_return")


def approx_normal_distribution(mean,std,sample_size = 1000):
    N = Normal('',mean,std)
    return [Kondisi(i,1,sample(N)) for i in range(0,sample_size)]

def normalisasi_peluang(list_kondisi) :
    total = sum([kondisi.Pr for kondisi in list_kondisi])
    return [Kondisi(kondisi.Nama,kondisi.Pr/total,kondisi.Total_return) for kondisi in list_kondisi]

def leverage (list_kondisi, leverage_rate):
    return [Kondisi(kondisi.Nama,
                    kondisi.Pr,
                    kondisi.Total_return * leverage_rate - leverage_rate + 1) 
            for kondisi in list_kondisi]

def multiplicative_performance (list_kondisi,r):
    p = 0
    for kondisi in list_kondisi:
        pf = (1 - r) + (r * kondisi.Total_return)
        p += kondisi.Pr*log(pf)
    return p

def additive_performance (list_kondisi,r):
    p = 0
    for kondisi in list_kondisi:
        pf = kondisi.Total_return
        p = p + kondisi.Pr*pf
    return p

def analysis (list_kondisi,performance = multiplicative_performance) :
    r = symbols('r',real = True)
    list_kondisi = normalisasi_peluang(list_kondisi)
    p = performance(list_kondisi,r)
    r_opt = solve(simplify(p.diff(r)),r)

    print('Optimal R is {r}'.format(r = r_opt))

    print('Expected return is {r} for every bet'.format(r = [simplify(p.subs(r,e)).evalf() for e in r_opt]))
    print('Expected return with 100% R is {r} for every bet'.format(r = simplify(p.subs(r,1)).evalf()))
    print('Expected return with 0% R is {r} for every bet'.format(r = simplify(p.subs(r,0)).evalf()))
    
def fast_step_analysis (list_kondisi,performance = multiplicative_performance, 
                        step = 0.1, start = 0, end = 1) :
    cur = start
    while cur < end:
        p = performance(list_kondisi,cur)
        print('Expected return with {r} R is {ret} for every bet'.format(r = cur,ret = p.evalf()))
        cur += step
    print('Expected return with {r} R is {ret} for every bet'.format(r = end,ret = p.evalf()))