'''
The script implements a forward simulator to generate synthetic data accepting 
hyper parameters (alpha,a0,a1) and a collapsed Gibbs sampler for this DPMM (see report).
Accuracy and Performances are also develeloped. Please define:
    
    - alpha            # Alpha hyper parameter
    - d0               # d0 Poisson distribution
    - a0               # Beta-distribution hyper parameter
    - a1               # Beta-distribution hyper parameter
    - s                # Number of points in d-b space
    - it               # Number of iterations for collapsed Gibbs sampling

D.Massaro & A.Mundo, KTH, Statistical Methods in Applied Computer Science FD2447/FD3447, 2021
 
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import scipy as sp
from collections import Counter 
from IPython import get_ipython

get_ipython().magic('reset -sf')




#------ Synthetic data generation functions -----------------------------------
def gen_dat(alpha, n, d0, a0, a1):
    np.random.seed(668)
    pi  = gem(alpha)
    Z_n = np.sort(list(map(lambda x: np.argmax(x), np.random.multinomial(1, pi, n))))
    
    theta_vect = np.random.beta(a0, a1, len(pi))
    # Generating d_n
    dn = np.random.poisson(d0, n)
    # Generating b_n 
    cnt = Counter(Z_n)  # Counter definition
    cc  = 0             # Starting counter
    bn  = np.array([])
    for i in range(len(pi)):
        if (cnt[i] != 0):
            bn = np.concatenate((bn, np.random.binomial(dn[cc:cnt[i] + cc], theta_vect[i], cnt[i])))
        cc +=cnt[i]
            
    return dn, bn, Z_n

def gem(alpha):
    # Stick breaking construction 
    pi = []
    residual = 1.0 # 1.0 stick
    while (residual > 0.00001):
        beta_k = np.random.beta(1, alpha, 1)[0] 
        pi.append(beta_k*residual)
        residual *= (1 - beta_k)
    pi.append(residual)
    return pi
#------------------------------------------------------------------------------ 


#---------- Collapsed Gibbs Sampler DPMM --------------------------------------
class DPMM():
    def __init__(self, data, alpha, a0, a1):
        self.a0    = a0
        self.a1    = a1
        self.N     = len(data) 
        self.alpha = alpha 
        self.z     = None 
        self.data  = data 
        self.starter() 
        self.n_sum = None 
        self.r_sum = None
        self.get_clusters()
        
    def get_clusters(self, ind_d=None):     # ind_d: data index
        if ind_d is not None:
            self.z[ind_d, :] = 0
            cnt_c = np.sum(self.z, axis=0)  # Column counter
            neq0_c = np.where(cnt_c > 0)[0] # Columns neq 0
            self.z = self.z[:, neq0_c]
            self.n_cl = self.z.shape[1]     # Number of classes
        self.counts = np.sum(self.z, axis=0)
        self.get_parameters()
 
    def get_parameters(self):
        self.n_sum = np.zeros((self.n_cl)) 
        self.r_sum = np.zeros((self.n_cl))
        for ind_cl in range(self.n_cl):
            ind_cls = self.z[:,ind_cl] >0
            self.n_sum[ind_cl] = np.sum(self.data[ind_cls,1]) 
            self.r_sum[ind_cl] = np.sum(self.data[ind_cls,0])
            
    def starter(self):
        z = np.zeros((self.N, 1)) 
        z[0] = 1
        for n in range(self.N):
            k = z.shape[1]
            prob = np.zeros(k + 1)
            prob[0:k] = np.sum(z, axis=0)
            prob[k] = self.alpha
            prob = prob / sum(prob)
            class_ind = np.random.choice(list(range(k + 1)), p=prob) 
            if class_ind == k:
                col = np.zeros((self.N, 1))
                z = np.append(z, col, axis=1)
            z[n, class_ind] = 1 
        self.z = z
        self.n_cl = z.shape[1]    
        
    def cGIBBSs(self, n_iter): # Collapsed Gibbs sampler
        for i in range(n_iter):
            self.sample()
                
    def sample(self):
        for i in range(self.N):
            self.get_clusters(ind_d=i)
            prob = self.perdictive_prob(i)
            cls_def = np.random.choice(list(range(len(prob))), p=prob) 
            z = self.z
            if cls_def == len(prob) - 1:
                col = np.zeros((self.N, 1))
                z = np.append(z, col, axis=1)
            z[i, cls_def] = 1
            cnt_cs = np.sum(z, axis=0) 
            nzc = np.where(cnt_cs > 0)[0] 
            z = z[:, nzc]
            self.z = z
            self.nClass = z.shape[1]
        self.get_clusters()
        return
    
    def perdictive_prob(self, ind_d):
        predictive = np.zeros(self.n_cl +1)
        for ind_cl in range(self.n_cl):
            n = self.n_sum[ind_cl] 
            r = self.r_sum[ind_cl]
            betabinom_log = st.betabinom.logpmf(self.data[ind_d][0],self.data[ind_d][1],1+r,self.alpha+n-r)
            predictive[ind_cl] = betabinom_log + np.log(self.counts[ind_cl]/(self.alpha+self.N-1)) 
        betabinom_log = st.betabinom.logpmf(self.data[ind_d][0], self.data[ind_d][1], 1, self.alpha) 
        predictive[self.n_cl] = betabinom_log +np.log(self.alpha/(self.alpha+self.N-1))
        prob = predictive - max(predictive) 
        prob = np.exp(prob)
        prob = prob / np.sum(prob)
        return prob
  
    def get_z(self):
        return self.z
    
    def rand_index(ref,z) :
        a=0
        b=0
        all_pairs = sp.special.binom(len(ref), 2) 
        for i in range(len(ref) - 1):
            for j in range(i + 1, len(ref)):
                if (z[i] == z[j] and ref[i] == ref[j]):
                    a += 1
                elif (z[i] != z[j] and ref[i] != ref[j]):
                    b += 1
        rand_index = (a + b) / all_pairs   
        return rand_index
#------------------------------------------------------------------------------              
    
                
#----------------- Performances Analysis --------------------------------------      
def iter_test(iter_vect, alpha, a0, a1, s, d0):
    rand_vect = []
    for it in iter_vect:
        dn, bn, Zn = gen_dat(alpha, s, d0=d0, a0=a0, a1=a1) 
        dataset = np.array(list(zip(bn, dn)))
        dpmm = DPMM(dataset, alpha, a0, a1)
        dpmm.cGIBBSs(n_iter=it)     
        z = dpmm.get_z()
        z = np.argmax(z, axis=1)
        rand_vect.append(DPMM.rand_index(Zn, z))
    return rand_vect  

def alpha_test(alpha_vect, a0, a1, it, s, d0): 
    rand_vect = []
    for a in alpha_vect:
        dn, bn, Zn = gen_dat(a, s, d0=d0, a0=a0, a1=a1) 
        dataset = np.array(list(zip(bn, dn)))
        dpmm = DPMM(dataset, a, a0, a1)
        dpmm.cGIBBSs(n_iter=it)
        z = dpmm.get_z()
        z = np.argmax(z, axis=1)
        rand_vect.append(DPMM.rand_index(Zn, z))
    return rand_vect

def s_test(s_vect, alpha, a0, a1, it, d0):
    rand_vect = []
    for s in s_vect:
        dn, bn, Zn = gen_dat(alpha, s, d0=d0, a0=a0, a1=a1) 
        dataset = np.array(list(zip(bn, dn)))
        dpmm = DPMM(dataset, alpha, a0, a1)
        dpmm.cGIBBSs(n_iter=it)     
        z = dpmm.get_z()
        z = np.argmax(z, axis=1)
        rand_vect.append(DPMM.rand_index(Zn, z))
    return rand_vect        
#------------------------------------------------------------------------------ 
        
        
        
        



if __name__ == '__main__':
    
    #------------------- FORWARD SIMULATOR ------------------------------------
    # Parameters 
    alpha = 1            # Alpha parameter
    d0    = 1000
    a0    = 1
    a1    = 1
    s     = 100          # Number of points in d-b space
    it    = 30           # Number of iterations for collapsed Gibbs sampling
    
    # Forward Simulator
    dn, bn, Zn = gen_dat(alpha, s, d0, a0, a1)
    # Combining dn-bn and corresponding labelling
    data = np.zeros([s,3])
    data[:,0] = dn; data[:,1] = bn; data[:,2] = Zn

    # Initial data scatter plot 
    for c in set(data[:,2]):
        data_c = data[data[:,2] == c]
        plt.scatter(data_c[:,0],data_c[:,1],label='cluster {}'.format(int(c)))
    
    plt.xlabel('$d^n$')
    plt.ylabel('$b^n$')
    plt.legend(loc='upper left')        
    plt.show()
    
    
    #------------------- CLUSTERING -------------------------------------------
    dataset = np.array(list(zip(bn, dn)))
    dpmm = DPMM(dataset, alpha, a0, a1)
    dpmm.cGIBBSs(n_iter=it)
       
    z = dpmm.get_z()
    z = np.argmax(z, axis=1)

    rand = DPMM.rand_index(z,Zn)
    print('Rand index: ',rand)
    
    # Combining dn-bn and corresponding labelling
    data = np.zeros([s,3])
    data[:,0] = dn
    data[:,1] = bn
    data[:,2] = z    
    # Initial data clusters plot 
    for c in set(data[:,2]):
        data_c = data[data[:,2] == c]
        plt.scatter(data_c[:,0],data_c[:,1],label='cluster {}'.format(int(c)))
        
    
    plt.xlabel('$d^n$')
    plt.ylabel('$b^n$')
    #plt.legend(loc='upper left')         
    plt.show()
    
  
    #------------------- ACCURACY ANALYSIS ------------------------------------
    # alpha_vect = np.array([1,2,3,5,7,10,15,18,20])
    # vect_alpha = alpha_test(alpha_vect, a0, a1, it, s, d0)

    # plt.plot(alpha_vect,vect_alpha)
    # plt.xlabel('$alpha$')
    # plt.ylabel('$rand_i$')      
    # plt.show()
    

    # s_vect =np.array([50, 100,150,250,350,450,600,700,850,1000])
    # vect_size = alpha_test(s_vect, alpha, a0, a1, it, d0)
    
    # plt.plot(s_vect,vect_size)
    # plt.xlabel('number of data points')
    # plt.ylabel('$rand_i$')      
    # plt.show()
    
    iter_vect =np.array([ 5, 10, 20, 30, 40 , 50, 100])
    vect_iter = alpha_test(iter_vect, alpha, a0, a1, s, d0)


    plt.plot(iter_vect,vect_iter)
    plt.xlabel('number of iterations')
    plt.ylabel('$rand_i$')      
    plt.show()




 

    


    




    