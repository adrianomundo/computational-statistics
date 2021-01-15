source("../2_2/BPF.R")

# CSMC function
csmc = function(x, y, sigma, beta, num_particles, steps) {
  
  particles = matrix(0, nrow = steps, ncol = num_particles)
  w = matrix(0, nrow = steps, ncol = num_particles)
  w.norm = matrix(0, nrow = steps , ncol = num_particles)
  ancestor = matrix(0, nrow = steps, ncol = num_particles)
  
  # initial step, t=1
  particles[1, ] = rnorm(num_particles, 0, sigma)
  particles[1, num_particles] = x[1] 
  ancestor[1, ] = 1:num_particles
  
  w[1, ] = dnorm(y[1], 0, sd = sqrt(beta^2 * exp(particles[1, ])), log = T)
  max.weight = max(w[1, ])
  w[1, ] = exp(w[1, ] - max.weight)
  w.norm[1, ] = w[1, ] / sum(w[1, ])
  
  log.likelihood = 0
  log.likelihood = log.likelihood + max.weight + log(sum(w[1, ])) - log(num_particles)
  
  for (t in 2:steps) {
    
    new.ancestors = multinomial.resampling(w.norm[t-1, ])
    new.ancestors[num_particles] = num_particles
    
    ancestor[t,] = ancestor[t-1, new.ancestors]
    
    particles[t, ] = rnorm(num_particles, particles[t-1, ancestor[t,]], sigma)
    particles[t, num_particles] =  x[t] 
    
    w[t, ] = dnorm(y[t], 0, sd = sqrt(beta^2 * exp(particles[t, ])), log = T)
    max.weight = max(w[t, ])
    w[t, ] = exp(w[t, ] - max.weight)
    w.norm[t, ] = w[t, ] / sum(w[t, ])
    
    log.likelihood = log.likelihood + max.weight + log(sum(w[t, ])) - log(num_particles)
  }
  
  end_steps = sample(1:num_particles, size = 1, prob = w.norm[steps, ])
  x.star = rep(0, steps)
  x.star[steps] = particles[steps, end_steps]
  
  for(t in steps:2){ 
    end_steps = ancestor[t, end_steps]
    x.star[t-1] =  particles[t-1, end_steps]
    
  }
  list(x.star = x.star, ancestors = ancestor, logl = log.likelihood)
  
}

# particle gibbs sampling
pg = function (y, sigma_square, beta_square, steps, num_particles, num_iteration) {
  
  initial.sigma_square = sigma_square
  initial.beta_square = beta_square
  
  bpf = bpf(y, 0.91, sqrt(sigma_square), sqrt(beta_square), num_particles, steps, multinomial.resampling)
  initial.x = bpf$x.hat
  
  result = csmc(initial.x, y, sqrt(initial.sigma_square), sqrt(initial.beta_square), num_particles, steps)
  initial.x = result$x.star
  
  sigma_square.seq = tibble(sigma_square = numeric())
  beta_square.seq = tibble(beta_square = numeric())
  sigma_square.seq  = add_row(sigma_square.seq, sigma_square = initial.sigma_square)
  beta_square.seq = add_row(beta_square.seq, beta_square = initial.beta_square)
  
  log.likelihood.seq = rep(NA, num_iteration)
  log.likelihood.seq[1] = bpf$logl
  
  for (k in 2:num_iteration) {
    
    initial.sigma_square = nimble::rinvgamma(1, shape = 0.01 + steps/2, scale = 0.01 + 0.5 * sum( ( initial.x[2:steps] - initial.x[1:(steps-1)] )^2))
    sigma_square.seq = add_row(sigma_square.seq, sigma_square = initial.sigma_square)
    
    result = csmc(initial.x, y, sqrt(initial.sigma_square), sqrt(initial.beta_square), num_particles, steps)
    
    initial.x = result$x.star
    initial.beta_square = nimble::rinvgamma(1, shape = 0.01 + steps/2, scale = 0.01 + 0.5 * sum(exp(-initial.x)*(y^2)))
    
    beta_square.seq = add_row(beta_square.seq, beta_square = initial.beta_square)
    
    log.likelihood.seq[k] = result$logl
    
    if (k %% 150 == 0) {
      cat(sprintf("Log-likelihood = %.4f \n", result$logl))
    }
  }
  
  list(sigma.density = sigma_square.seq$sigma_square, beta.density = beta_square.seq$beta_square, logl = log.likelihood.seq)
  
}
