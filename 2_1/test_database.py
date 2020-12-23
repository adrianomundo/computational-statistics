#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:01:59 2020

@author: daniele
"""
import numpy as np
import math as math
import operator as op
import functools
from scipy.special import loggamma
from collections import Counter
import matplotlib.pyplot as plt
from IPython import get_ipython


get_ipython().magic('reset -sf')

def estimate_pos(alphabet, data, sequence_id, prev_positions, alpha, alpha_prime, W):
    '''
    :return: a list containing a the posterior log-probability for each position to be the starting position
    of the sequence (sequence_id) : p(r_{sequence_id} | R_{sequence_id}, data)
    '''
    N = len(data)
    M = len(data[0])
    K = len(alphabet)
    # number of background positions
    B = N * (M - W)
    pos_proba = []

    for r_i in range(M - W):
        positions = list(prev_positions)
        positions[sequence_id] = r_i

        # Computing Bk for all k in alphabet
        background = [data[i][:positions[i]] + data[i][positions[i] + W:] for i in range(N)]
        counter_background = Counter(item for seq in background for item in seq)
           
        # Computing (Nkj for all k in alphabet) for all j in W
        words = [data[i][positions[i]:positions[i] + W] for i in range(N)]
        counter_words = [Counter([words[i][j] for i in range(N)]) for j in range(W)]
        
        # Normalizing constants
#        z_background =  float(math.gamma(math.log(sum(alpha_prime)))) / math.gamma(math.log(B+sum(alpha_prime)))
#        z_word       =  float(math.gamma(math.log(sum(alpha)))) / math.gamma(math.log(N+sum(alpha)))
        z_background =  float(loggamma((sum(alpha_prime)))) / loggamma((B+sum(alpha_prime)))
        z_word       =  float(loggamma((sum(alpha)))) / loggamma((N+sum(alpha)))
                                                             
    

        
        # background probabilities
        p_list = [float(math.gamma(counter_background[alphabet[k]] + alpha_prime[k]))/
                  float(math.gamma(alpha_prime[k])) for k in range(K)]
        p = z_background*functools.reduce(op.mul, p_list, 1)
    
        # magic word probabilities
        for j in range(W):
            p_list = [float(math.gamma(counter_words[j][alphabet[k]] + alpha[k]))/
                      float(math.gamma(alpha[k])) for k in range(K)]
            p_j = (z_word)*functools.reduce(op.mul, p_list, 1)
            p *= p_j
       

        pos_proba.append(p)

    return pos_proba

def r0_estimator(alphabet, data, alpha, alpha_prime, W, iterations, minimum, step):

    samples = [] 
    N = len(data)
    M = len(data[0])

    # Initial position r0' is randomly generated
    samples.append([np.random.randint(0, M-W+1) for i in range(N)])
    totsamples = np.zeros((N,iterations))
    
    for i in range(iterations):
        temp_r = []
        for j in range(N):
            
            p_pos = estimate_pos(alphabet, data, j, samples[-1], alpha, alpha_prime, W)
            
            # Normalization
            s = float(sum(p_pos))
            p_pos = [p / s for p in p_pos]

            #sample
            position = np.argmax(np.random.multinomial(1,p_pos))
            
            print(position)
            temp_r.append(position)
            totsamples[j,i] = position   
            
        samples.append(temp_r)
        
    # Plot convergence
    
    plt.figure(facecolor="white")
    aa = np.asarray(totsamples)
    plt.plot(aa[0,:],'.-')
    plt.title('$r_0$ convergence')
    plt.ylabel('$r_0$ ')
    plt.xlabel('number of iterations')
    plt.show()   
    
    # Select the iterations between minimum step and a given step
    samples = [samples[j] for j in range(minimum, iterations, step)]

    # The selected positions r0 are the ones most frequent during iterations
    return totsamples,[Counter([samples[j][i] for j in range(len(samples))]).most_common(1)[0][0] for i in range(N)]



if __name__ == '__main__':
    alphabet = ['0','1','2','3','4','5','6'] # 0-6 
    alpha_prime = np.ones(7)
    alpha = np.ones(7)*9.000000000000000222e-01
    N=10
    M=40
    W=6
    K=7

    # Generate sequences and true starting positions
    data0 = np.loadtxt('D.txt', delimiter='\t') 
    dd= data0[0:N,:]
    data = dd.tolist()
    
    # Estimates the starting positions
    iterations = 100
    minstep = 50
    steplength = 10
    
 
    tot, r_initial = r0_estimator(alphabet, data, alpha, alpha_prime, W,iterations,minstep,steplength)
    
    print(r_initial)
    
    # Mean and Variance computation for convergency analysis 
    for i in range(N):
        print("Mean Sequence {}: {}".format(i+1,np.mean(tot[i,:])))   
        print("Std Sequence {}: {}".format(i+1,np.std(tot[i,:])))   
        print('Number of Occurences in the entire sequence: {} - {}%'.format(np.count_nonzero(tot[i,:] == r_initial[i]),
              np.count_nonzero(tot[i,:] == r_initial[i])/iterations*100))
        print('Number of Occurences in the after burn-in: {} - {}%'.format(np.count_nonzero(tot[i,minstep:-1] == r_initial[i]),
              np.count_nonzero(tot[i,minstep:-1] == r_initial[i])/(iterations-minstep)*100))
        
    plt.figure(facecolor="white")
    plt.plot(tot[-1,:],'.-')
    plt.title('$r_0$ convergence')
    plt.ylabel('$r_0$ ')
    plt.xlabel('number of iterations')
    plt.show()   