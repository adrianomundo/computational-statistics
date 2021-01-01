"""
Convergence analysis for a given set of parameters and changing the overall number of iterations.

D.Massaro & A.Mundo, KTH, Statistical Methods in Applied Computer Science FD2447/FD3447, 2020

"""
import numpy as np
import math as math
import operator as op
import functools
import Gibbs_gen as gen
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
        z_background =  float(math.gamma(math.log(sum(alpha_prime)))) / math.gamma(math.log(B+sum(alpha_prime)))
        z_word       =  float(math.gamma(math.log(sum(alpha)))) / math.gamma(math.log(N+sum(alpha)))
        
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
    a = 0 # accuracyy counter
 
    np.random.seed(12345)
    # Initial position r0' is randomly generated
    samples.append([np.random.randint(0, M-W+1) for i in range(N)])
    
    for i in range(iterations):
        temp_r = []
        for j in range(N):
            
            p_pos = estimate_pos(alphabet, data, j, samples[-1], alpha, alpha_prime, W)
            
            # Normalization
            s = float(sum(p_pos))
            p_pos = [p / s for p in p_pos]
    
            #sample
            position = np.argmax(np.random.multinomial(1,p_pos))
            
            temp_r.append(position)
            
            if (i>minimum):
                if (position==positions[j]):
                    a = a+1
        hacc = a/(iterations-minimum)/N # historical accuracy
        #print(hacc)
                    
                    
            
        samples.append(temp_r)

    
    # Select the iterations between minimum step and a given step
    samples = [samples[j] for j in range(minimum, iterations, step)]

    # The selected positions r0 are the ones most frequent during iterations
    return [Counter([samples[j][i] for j in range(len(samples))]).most_common(1)[0][0] for i in range(N)], hacc



if __name__ == '__main__':
    alphabet = ['2','3','5','7'] # 1-digit prime numbers176785 -1613.95723659]
    alpha_prime = [0.1,0.1,0.1,0.1]
    alpha = [9,7,20,2]
    W = 10    # Magic word length
    N = 5    # Number of sequences
    M = 30   # Length of the sequence

    # Generate sequences and true starting positions
    data, positions = gen.generate_sequences(alphabet, alpha, alpha_prime, N, M, W)
    
    # Estimates the starting positions
    
    it_vect = np.linspace(100,1000,10)
    minstep = 50
    steplength = 10
    his_acc = []
    for iit in it_vect:    
 
       iit = int(iit)
       r_initial, hisacc = r0_estimator(alphabet, data, alpha, alpha_prime, W,iit,minstep,steplength)
    
       his_acc.append(hisacc)
       
    # Plot convergence
    plt.figure(facecolor="white")
    plt.plot(it_vect,his_acc,'.-')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.show()   
    
       
       
       
     
       