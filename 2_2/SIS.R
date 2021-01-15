# Sequential importance sampling for the stochastic volatility model
# N = number of particles
# T = number of time steps

sis = function(y, N, T, phi, sigma, beta) {
  
  x.sis = matrix(0, nrow = T, ncol = N) 
  w.sis = matrix(0, nrow = T, ncol = N)
  w.norm.sis = matrix(0, nrow = T, ncol = N)
  
  # Initial state, t = 1
  x.sis[1, ] = rnorm(N, mean = 0, sd = sigma) 
  
  # weighting step, t = 1
  w.sis[1, ] = dnorm(y[1], mean = 0, sd = sqrt(beta^2 * exp(x.sis[1, ]))) 
  w.norm.sis[1, ] = w.sis[1, ] / sum(w.sis[1, ])
  
  x.hat.sis = rep(NA, T) 
  x.hat.sis[1] = sum(w.norm.sis[1, ] * x.sis[1,])
  
  for(t in 2:T) {
    x.sis[t, ] = rnorm(N, mean = 1.00 * x.sis[t-1, ], sd = sigma)
    w.sis[t, ] = dnorm(y[t], mean = 0, sd = sqrt(beta^2 * exp(x.sis[t, ]))) * w.sis[t-1, ]
    
    w.norm.sis[t, ] = w.sis[t, ] / sum(w.sis[t, ])
    x.hat.sis[t] = sum(w.norm.sis[t, ] * x.sis[t, ])
  }
  
  return(list(x.hat = x.hat.sis, wnorm = w.norm.sis))
}