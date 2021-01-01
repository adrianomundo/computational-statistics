"""
The script generates N sequences of length M over an alphabet of length W, 
each containing a "magic" word of length W. The "magic letters" are sampled from
a categorical distribution Cat(x|theta_j) where theta_j has a prior Dir(theta|alpha).
The "background letters" are sampled from a categorical distribution Cat(x|theta) 
where theta has a prior Dir(theta|alpha').

- W  (magic word length)
- N  (number of sequences)
- M  (length of the sequence)

D.Massaro & A.Mundo, KTH, Statistical Methods in Applied Computer Science FD2447/FD3447, 2020

"""

import numpy as np
from IPython import get_ipython

get_ipython().magic('reset -sf') # To reset variables


# FUNCTIONS DEFINITION
def sample_cat(alphabet, categorical):
    """
    Sampling from Categorical Distribution with "cetegorical" variable from
    given "alphabet"
    - alphabet: the letters of the sequences
    - categorical: theta_j or theta distribution
    - output: a sample from the given categorical distribution
    """
    return alphabet[np.argmax(np.random.multinomial(1,categorical))]

def generate_sequences(alphabet, alpha, alpha_prime,N,M,W):
    """
    Built up a list of sequences and a list of starting positions
    """
    # Background disitribution
    theta_back = np.random.dirichlet(alpha_prime)   
    sequences = [[sample_cat(alphabet, theta_back) for j in range(M)]for i in range(N)]

    # Magic word generation
    thetas_mw = np.random.dirichlet(alpha, W) # theta magic word
    positions = [np.random.randint(0, M-W+1) for i in range(N)] # sampled uniformly
    

    sequences = [[sample_cat(alphabet, thetas_mw[x-pos]) if x >= pos and x < pos+W # magic word loc
                  else seq[x] for x in range(len(seq))] # back letters loc
                  for seq, pos in zip(sequences, positions)]

    return sequences, positions





# MAIN
if __name__ == "__main__":
    # Parameters
    W = 3
    N = 5
    M = 10
    alphabet = ['2','3','5','7'] # 1-digit prime numbers
    alpha_prime = [1,1,1,1]  
    alpha       = [1,7,12,2]

    # Set random seed
    np.random.seed(12345)

    # Generate sequences
    seqs, pos = generate_sequences(alphabet,alpha, alpha_prime,N,M,W)
        
    print("Sequences: ")
    for s in seqs :
        print(s)

    print("Positions", pos)
