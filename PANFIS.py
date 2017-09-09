# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 21:43:07 2016
PANFIS
@author: calvin
"""

import numpy as np

# GENEFIS without rule merging and input pruning
# Can only get one output, see MOGENEFIS for more
class GENEFIS:    
    def __init__(self,n_input):
        self.n_input = n_input
        self.g = np.power(10,-n_input)
        self.k_err = 0.1*self.g
        #self.rho_b = np.power(10,-n_input)
        self.rho_b = 0.01
        self.rho_a = np.exp(-1)
        self.k_in = np.power(10, -n_input/2)
        # Number of trained data
        self.n = 0
        # Number of rules
        self.r = 0
        # Center vector for rules, length = n_input
        self.center = []
        # Dispersion matrix for rules, shape = n_input x n_input
        self.inv_dispersion = []
        # How many times a rule win
        self.N = []
        # weight in a rule, a vector of length 1 + n_input
        self.w = []
        # Update weight matrix , Eq 55
        self.P = []
        self.mean_error = 0
        self.error_variance_squared = 0.0000000000000001
        self.forget_factor = 1
        
    def train(self,x,d):
        x = np.array(x)
        target = d[0]
        self.n += 1
        if (self.r == 0):
            # make new rule
            self.r += 1
            self.center.append(x)
            self.inv_dispersion.append(np.eye(self.n_input))
            self.N.append(1)
            self.w.append(np.hstack(([target],np.zeros(self.n_input))))
            self.P.append(np.eye(self.n_input + 1) * 1000000000000)
        else:
            #Phase 1, Growth and adaptation of fuzzy rules
            r = self.r
            error = np.abs(self.output(x)[0]-target)
            n = self.n
            old_mean_error = self.mean_error
            old_variance_squared = self.error_variance_squared
            new_mean_error = old_mean_error + (error - old_mean_error) / n
            new_variance_squared = (n-1)*old_variance_squared/n + np.square(error-old_mean_error)/n
            self.mean_error = new_mean_error
            self.error_variance_squared = new_variance_squared
            # Compute posteriors
            Vs = [self.V(i) for i in range(r)]
            Rs = [self.R(x,i) for i in range(r)]
            likelihoods = [R/np.sqrt(2*np.pi*V) for R,V in zip(Rs,Vs)]
            total_prior = np.sum(self.N)
            priors = [self.N[i]/total_prior for i in range(r)]
            lps = [l*p for l,p in zip(likelihoods,priors)]
            total_posterior = np.sum(lps)
            posteriors = [lp/total_posterior for lp in lps]
            
            Vmax = self.rho_b * np.sum(Vs)
            p_and_i = zip(posteriors,range(r))
            Pwin,iwin = max(p_and_i)
            Vwin = Vs[iwin]
            Rwin = Rs[iwin]
            rho_a = self.rho_a
            if (Rwin >= rho_a and Vwin <= Vmax):
                self.RuleUpdate(iwin,x)
            elif (Rwin < rho_a and Vwin > Vmax):
                # Check DS criterion
                if (new_mean_error + new_variance_squared -old_mean_error - old_variance_squared > 0):
                    new_center = x
                    distances = [np.abs(new_center - center) for center in self.center]
                    max_d = np.max(distances,axis=0) + 0.000000001
                    e = 0.5
                    z = np.sqrt(np.log(1/e))
                    new_inv_dispersion = np.diag(z*max_d)
                    dets = [1/np.linalg.det(i_d) for i_d in self.inv_dispersion]
                    new_det = 1/np.linalg.det(new_inv_dispersion)
                    pow_to_k= np.vectorize(lambda x:np.power(x,self.n_input))
                    DS = pow_to_k(new_det)/np.sum([pow_to_k(det) for det in dets + [new_det]])
                    if (DS >= self.g):
                        # Add the rule
                        self.r += 1
                        self.center.append(new_center)
                        self.inv_dispersion.append(new_inv_dispersion)
                        self.N.append(1)
                        self.w.append(self.w[iwin])
                        self.P.append(self.P[iwin])
            elif(Rwin >= rho_a and Vwin > Vmax):
                self.center[iwin] = x
                while (self.V(iwin) > Vmax):
                    self.inv_dispersion[iwin] *= 1/0.98
            else:
                self.RuleUpdate(iwin,x)
            # Phase 2 prune inactive rules
            r = self.r
            pow_to_k= np.vectorize(lambda x:np.power(x,self.n_input))
            dets = [1/np.linalg.det(i_d) for i_d in self.inv_dispersion]
            Einf = [pow_to_k(d) for d in dets]
            Einf = np.array(Einf)/np.sum(Einf)
            to_be_deleted = [i for E,i in zip(Einf,range(r)) if E <= self.k_err]
            
            for to_delete in reversed(to_be_deleted):
                self.r -= 1
                del self.center[to_delete]
                del self.inv_dispersion[to_delete]
                del self.N[to_delete]
                del self.w[to_delete]
                del self.P[to_delete]
                
            # Adjust weight
            r = self.r
            n_input = self.n_input
            for i in range(r):
                # Use recursive consquent learning in generalized smart efs because
                # the original paper is unintellligible
                old_P = self.P[i]
                old_w = self.w[i]
                eX = np.matrix(self.extend_input(x)).T
                R = self.R(x,i)
                if (R == 0):
                    gamma = 0
                else:
                    gamma = old_P * eX / (self.forget_factor/np.sqrt(R)+eX.T*old_P*eX)      
                new_w = old_w + np.asarray(gamma).flatten()*np.asscalar(d-old_w*eX)
                new_P = (1/self.forget_factor) * (np.eye(n_input + 1) - gamma*eX.T) * old_P
                
                self.P[i] = new_P
                self.w[i] = new_w
            
    def V(self,i):
        return 1/(np.linalg.det(self.inv_dispersion[i]))
            
        
    # Firing strength given input x and rule i
    # Eq 1
    def R(self,X,i):
        Ci = self.center[i]
        sigma_inv = self.inv_dispersion[i]
        h = np.matrix(X-Ci)
        # According to genefis.m that dr mahardhika share
        # Funny that in ther paper /2 do not exist
        return max(np.exp(-h*sigma_inv*h.T*0.5),0.000000001)
    
    def extend_input(self,X):
        return np.concatenate(([1], np.array(X)))
    
    # Eq 2
    def output(self,X):
        Rs = [self.R(X,i) for i in range(self.r)]
        ws = self.w
        eX = self.extend_input(X)
        up = np.sum([Ri*np.sum(wi*eX) for Ri,wi in zip(Rs,ws)])
        down = np.sum(Rs)
        return [float(up/down)]
        
    def RuleUpdate(self,index_win,X):        
        win_rule = index_win
        
        old_center = self.center[win_rule]
        old_inv_dispersion = self.inv_dispersion[win_rule]
        old_N = self.N[win_rule]
        new_N = old_N + 1
        alpha = (1/new_N)
        new_center = (old_N * old_center + (X - old_center))/new_N
        z = old_inv_dispersion * np.matrix(X-new_center).T
        h = np.matrix(X-old_center)
        new_inv_dispersion = (old_inv_dispersion) / (1 - alpha) + \
                            (alpha/(1-alpha)) * z * z.T  / \
                            (1 + alpha*h*old_inv_dispersion*h.T)
        
        self.center[win_rule] = new_center
        self.inv_dispersion[win_rule] = new_inv_dispersion
        self.N[win_rule] = new_N
        
class MOGENEFIS:    
    def __init__(self,n_input,n_output):
        self.machines = [GENEFIS(n_input) for i in range(n_output)]
        
    def train(self,x,d):
        for t,m in zip(d,self.machines):
            m.train(x,[t])
    
    def output(self,x):
        return [m.output(x)[0] for m in self.machines]

# Generalized smart evolving fuzzy system
class GSEFS:    
    def __init__(self,n_input,fac = 0.0001,density_impact_degree = 4):
        self.n_input = n_input
        self.fac = fac
        self.density_impact_degree = density_impact_degree
        # Number of rules
        self.r = 0
        # Center vector for rules, length = n_input
        self.center = []
        # Dispersion matrix for rules, shape = n_input x n_input
        self.inv_dispersion = []
        # weight in a rule, a vector of length 1 + n_input
        self.w = []
        # Number of sample a rule win so far
        self.N = []
        self.P = []
        self.forget_factor = 1
        
    def train(self,x,d):
        x = np.array(x)
        x =  x + np.random.random(self.n_input)*0.000001
        target = d[0]
        if (self.r == 0):
            # Make new rule
            self.r += 1
            self.center.append(x)
            self.inv_dispersion.append(np.eye(self.n_input))
            self.N.append(1)
            self.w.append(np.hstack(([target],np.zeros(self.n_input))))
            self.P.append(np.eye(self.n_input + 1) * 1000000000000)
        else:
            r = self.r
            #Eq 10, search minimal rule distance
            mahal_and_i = [(self.mahalanobis(x,i),i) for i in range(r)]
            mahal_win,iwin = min(mahal_and_i)
            vigilance_threshold = self.fac * np.power(self.n_input,np.sqrt(2))/ \
                                np.power((1-1/(self.N[iwin]+1)),self.density_impact_degree)
            if (mahal_win > vigilance_threshold):
                # 3.1.1 a new rule evolved
                # Make new rule
                self.r += 1
                self.center.append(x)
                self.inv_dispersion.append(sum(self.inv_dispersion)/(self.r-1))
                self.inv_dispersion[0] = np.eye(self.n_input) * self.fac
                self.N.append(1)
                self.w.append(self.w[iwin])
                self.P.append(self.P[iwin])
                iwin = self.r - 1
            else:
                win_rule = iwin
                X = x
                old_center = self.center[win_rule]
                old_inv_dispersion = self.inv_dispersion[win_rule]
                old_N = self.N[win_rule]
                new_N = old_N + 1
                alpha = (1/new_N)
                new_center = (old_N * old_center + (X - old_center))/new_N
                z = old_inv_dispersion * np.matrix(X-new_center).T
                new_inv_dispersion = (old_inv_dispersion) / (1 - alpha) - \
                                    (alpha/(1-alpha)) * z * z.T  / \
                                    (1 + alpha*np.square(self.mahalanobis_dist(X,old_center,old_inv_dispersion)))
                self.center[win_rule] = new_center
                self.inv_dispersion[win_rule] = new_inv_dispersion
                self.N[win_rule] = new_N
            # Merge Rules
            # Check condition Eq 23
            # Eq 17
            def olap(c_win,c_k,sigma_inv_win,sigma_inv_k):
                sigma_inv = (sigma_inv_win + sigma_inv_k)/2
                mahal_win_k = self.mahalanobis_dist(c_win,c_k,sigma_inv)
                det = np.linalg.det
                ret = np.square(mahal_win_k)/8 + np.log(det(sigma_inv)/np.sqrt(det(sigma_inv_win)*det(sigma_inv_k)))/2
                return ret
            # Eq 19
            def s_cons(w_win,w_k):
                a = np.hstack((w_win,[-1]))
                b = np.hstack((w_k,[1]))
                c = np.sum(a*b)
                d = np.linalg.norm(a)*np.linalg.norm(b)
                phi = c/d
                return phi/np.pi
            
            center_win = self.center[iwin]
            sigma_inv_win = self.inv_dispersion[iwin]
            N_win = self.N[iwin]
            w_win = self.w[iwin]
            P_win = self.P[iwin]
            rule_to_be_deleted = []
            for i in range(self.r):
                if (i == iwin):
                    continue
                center_k = self.center[i]
                sigma_inv_k = self.inv_dispersion[i]
                
                # Condition 23, conventional
                if (olap(center_win,center_k,sigma_inv_win,sigma_inv_k) >= 0.8):
                    # Will be merged
                    rule_to_be_deleted.append(i)
                    w_k = self.w[i]
                    N_k = self.N[i]
                    P_k = self.P[i]
                    new_N = N_win + N_k
                    new_center = (N_win*center_win + N_k*center_k)/new_N
                    less_supported_center= None
                    if (N_k < N_win):
                        less_supported_center = center_k
                    else:
                        less_supported_center = center_win
                    diff_center = np.matrix(new_center - less_supported_center)
                    magic_diag = 1/(diff_center.T*diff_center)
                    finish_diag = np.diag(np.diag(magic_diag))
                    new_sigma_inv = N_win*sigma_inv_win + N_k * sigma_inv_k
                    new_sigma_inv += N_win*N_k*finish_diag/new_N
                    new_sigma_inv /= new_N
                    center_win = new_center
                    sigma_inv_win = new_sigma_inv
                    N_win = new_N
                    
                    def new_w_and_p():
                        w_w,c_w,si_w,n_w,P_w = (w_win,center_win,sigma_inv_win,N_win,P_win)
                        w_s,c_s,si_s,n_s,P_s = (w_k,center_k,sigma_inv_k,N_k,P_k)
                        if (N_win > N_k):
                            w_w,c_w,si_w,n_w,P_w = (w_k,center_k,sigma_inv_k,N_k,P_k)
                            w_s,c_s,si_s,n_s,P_s = (w_win,center_win,sigma_inv_win,N_win,P_win)
                        alpha = n_s / (n_s+n_w)
                        s_rule = olap(c_s,c_w,si_s,si_w)
                        s_con = s_cons(w_s,w_w)
                        cons = np.asscalar(np.exp(-np.square(s_rule/s_con - 1) / np.power(1/s_rule,7)))
                        w_new = w_s + alpha * cons * (w_s - w_w)
                        P_new = P_s + alpha * cons * (P_s - P_w)
                        return w_new,P_new
                    w_win,P_win = new_w_and_p()
            # Assign new param
            self.center[iwin] = center_win
            self.inv_dispersion[iwin] = sigma_inv_win
            self.N[iwin] = N_win
            self.w[iwin] = w_win
            self.P[iwin] = P_win
            # delete rule
            for i in sorted(rule_to_be_deleted,reverse=True):
                self.r -= 1
                del self.center[i]
                del self.inv_dispersion[i]
                del self.N[i]
                del self.w[i]
                del self.P[i]
            r = self.r
            n_input = self.n_input
            for i in range(r):
                # Use recursive consquent learning in generalized smart efs because
                # the original paper is unintellligible
                old_P = self.P[i]
                old_w = self.w[i]
                eX = np.matrix(self.extend_input(x)).T
                R = self.R(x,i)
                if (R == 0):
                    gamma = 0
                else:
                    gamma = old_P * eX / (self.forget_factor/np.sqrt(R)+eX.T*old_P*eX)      
                new_w = old_w + np.asarray(gamma).flatten()*np.asscalar(d-old_w*eX)
                new_P = (1/self.forget_factor) * (np.eye(n_input + 1) - gamma*eX.T) * old_P
                
                self.P[i] = new_P
                self.w[i] = new_w
            
        
    # Mahalanobis distance of rule i and x
    def mahalanobis(self,X,i):
        Ci = self.center[i]
        sigma_inv = self.inv_dispersion[i]
        return self.mahalanobis_dist(X,Ci,sigma_inv)
        
    # Mahalanobis distance of vector X and vector Y with dispersion matrix inverted (Sigma inv)
    def mahalanobis_dist(self,X,Y,sigma_inv):
        h = np.matrix(X-Y)
        ret = np.asscalar(np.sqrt(h*sigma_inv*h.T))
        return ret
        
    # Firing strength given input x and rule i
    # Eq 3
    def R(self,X,i):
        return max(np.exp(-np.square(self.mahalanobis(X,i))/2),0.000000000001)
    
    def extend_input(self,X):
        return np.concatenate(([1], np.array(X)))
    
    # Eq 4
    def output(self,X):
        Rs = [self.R(X,i) for i in range(self.r)]
        ws = self.w
        eX = self.extend_input(X)
        up = np.sum([Ri*np.sum(wi*eX) for Ri,wi in zip(Rs,ws)])
        down = np.sum(Rs)
        return [up/down]
        
class MOGSEFS:    
    def __init__(self,n_input,n_output,fac,density_impact_degree = 4):
        self.machines = [GSEFS(n_input,fac,density_impact_degree) for i in range(n_output)]
        
    def train(self,x,d):
        for t,m in zip(d,self.machines):
            m.train(x,[t])
    
    def output(self,x):
        return [m.output(x)[0] for m in self.machines]