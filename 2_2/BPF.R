# Bootstrap particle filter
# N = number of particles
# T = number of T steps

bpf = function(y, phi, sigma, beta, N, T, resampling.function) {
  
  particles = matrix(0, nrow = T, ncol = N)
  x.hat = matrix(0, nrow= T, ncol = 1)
  ancestors = matrix(1, nrow = T, ncol = N)
  
  w.bpf = matrix(1, nrow = T, ncol = N)
  w.norm.bpf = matrix(1, nrow = T, ncol = N)
  
  # initial step, t=1
  particles[1, ] = rnorm(N, 0, sigma)
  ancestors[1, ] = 1:N
  
  # weighting and propagation step, t=1
  w.bpf[1, ] = dnorm(y[1], mean = 0, sd = sqrt(beta^2 * exp(particles[1, ])), log = T)
  max.weight = max(w.bpf[1, ])
  w.bpf[1, ] = exp(w.bpf[1, ] - max.weight)
  sum_log_w.bpf = sum(w.bpf[1, ])
  
  w.norm.bpf[1, ] = w.bpf[1, ] / sum_log_w.bpf
  
  log.likelihood = 0
  log.likelihood = log.likelihood + max.weight + log(sum_log_w.bpf) - log(N)
  
  x.hat[1, 1] = sum(w.norm.bpf[1, ] * particles[1, ])
  
  for(t in 2:T){
    new.ancestors = resampling.function(w.norm.bpf[t-1, ])
    particles[t, ] = rnorm(N, mean = 1.0*particles[t-1, new.ancestors], sigma) 
    
    w.bpf[t, ] = dnorm(y[t], 0, sqrt(beta^2 * exp(particles[t, ])), log = TRUE)
    max.weight = max(w.bpf[t, ]) 
    w.bpf[t, ] = exp(w.bpf[t, ] - max.weight)
    sum_log_w.bpf = sum(w.bpf[t, ])
    
    w.norm.bpf[t, ] = w.bpf[t, ] / sum_log_w.bpf
    
    log.likelihood = log.likelihood + max.weight + log(sum_log_w.bpf) - log(N)
    
    x.hat[t, 1]  = sum(w.norm.bpf[t, ] * particles[t, ])
  }
  
  ancestor.index  = sample(1:N, size=1, prob = w.norm.bpf[T, ])
  x.hat.filtered = particles[, ancestor.index]
  
  return(list(x.hat= x.hat, x.hat.filtered = x.hat.filtered, w.norm.bpf = w.norm.bpf, logl = log.likelihood))
}

# Resampling functions

multinomial.resampling = function(w.vector, N = NA){
  # generate N values between 0 and 1
  N = ifelse(is.na(N), length(w.vector))
  N = length(w.vector)
  u = rep(NA, N)
  u.tilda = runif(N)^(1/(1:N))
  u[N] = u.tilda[N] 
  k = N - 1
  
  while(k > 0){
    u[k] = u[k + 1] * u.tilda[k]
    k = k-1
  }
  
  sampled = rep(NA, N)
  total = 0
  i = 1
  j = 1
  
  while(j <= N && i <= N){
    total = total + w.vector[i]
    while(j <= N && total > u[j]){
      sampled[j] = i
      j = j + 1
    }
    i = i + 1
  }
  sampled
} 

stratified.resampling = function(w.vector, N = NA) {
  N = ifelse(is.na(N), length(w.vector))
  sampled = rep(NA, N)
  counting = w.vector[1]
  j = 1
  
  for(i in 1:N){
    u = (runif(1) + i-1) / N
    while(counting < u){
      j = j + 1
      counting = counting + w.vector[j]
    }
    sampled[i] = j
  }
  sampled
}
