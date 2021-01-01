"""
The script generates reads the database with N sequences of length M over an alphabet of length W, 
each containing a "magic" word of length W. 
The initial positions are estimated for a given set of paramateres:

- W  (magic word length)
- N  (number of sequences)
- M  (length of the sequence)
- iterations (number of iterations)
- minstep (lag between samples)
- steplength (number of burn-in iterations)

D.Massaro & A.Mundo, KTH, Statistical Methods in Applied Computer Science FD2447/FD3447, 2020

"""

import numpy as np
import math as math
import operator as op
import functools
from scipy.special import loggamma
from collections import Counter
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

    np.random.seed(12345)
    
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
            
            #print(position)
            temp_r.append(position)
            totsamples[j,i] = position   
            
        samples.append(temp_r)
        

    
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
    iterations = 400
    minstep = 200
    steplength = 10
    
 
    tot, r_initial = r0_estimator(alphabet, data, alpha, alpha_prime, W,iterations,minstep,steplength)
    
    print(r_initial)
    
