library(tibble)
library(dplyr)
library(ggplot2)
library(magrittr)
library(purrr)
library(invgamma)

source("../2_2/BPF.R")
source("../2_2/SV_data_generator.R")

pmh = function(y, sigma, beta, num_particles, T, num_iteration, step, resampling.function){
  
  theta = matrix(0, nrow = 2, ncol = num_iteration)
  theta.proposal = matrix(0, nrow = 2, ncol = num_iteration)
  
  log.likelihood = matrix(0, nrow = 1, ncol = num_iteration)
  log.likelihood.proposal = matrix(0, nrow = 1, ncol = num_iteration)
  proposal.accepted = matrix(0, nrow = 1, ncol = num_iteration)
  
  # initial values
  theta[1, 1] = sigma
  theta[2, 1] = beta
  
  results = bpf(y, 0.91, sigma = sqrt(theta[1, 1]), beta = sqrt(theta[2, 1]), num_particles, T, resampling.function)
  log.likelihood[1, 1] = results$logl 
  
  for(k in 2:num_iteration) {
    
    theta.proposal[1, k]= max(0.0001, rnorm(1, theta[1, k - 1], step[1]))
    theta.proposal[2, k]= max(0.01, rnorm(1, theta[2, k - 1], step[2]))
    
    results = bpf(y, 0.91, sigma = sqrt(theta.proposal[1, k]), beta = sqrt(theta.proposal[2, k]), num_particles, T, resampling.function)
    log.likelihood.proposal[1, k] = results$logl
    
    prior.sigma = nimble::dinvgamma(theta.proposal[1, k], shape = 0.01, scale = 0.01, log = TRUE)
    difference.sigma = prior.sigma - nimble::dinvgamma(theta[1, k-1], shape = 0.01, scale = 0.01, log = TRUE)
    
    prior.beta = nimble::dinvgamma(theta.proposal[2, k], shape = 0.01, scale = 0.01, log = TRUE)
    difference.beta = prior.beta - nimble::dinvgamma(theta[2, k-1], shape = 0.01, scale = 0.01, log = TRUE)
    
    prior.difference.sum = difference.sigma + difference.beta
    
    likelihood.difference = log.likelihood.proposal[1, k] - log.likelihood[1, k - 1]
    acceptance.probability = exp(prior.difference.sum + likelihood.difference)
    
    uniform = runif(1)
    if (uniform <= acceptance.probability) { 
      theta[1:2, k] = theta.proposal[1:2, k] 
      log.likelihood[1, k] = log.likelihood.proposal[1, k]
      proposal.accepted[1, k] = 1
    } 
    else {
      theta[1:2, k] = theta[1:2, k-1] 
      log.likelihood[1, k] = log.likelihood[1, k - 1]
      proposal.accepted[1, k] = 0
    }
    if (k %% 50 == 0) {
      cat(sprintf("Posterior Mean - sigma and beta: %f %f \n", mean(theta[1, 1:k]),mean(theta[2, 1:k])))
    }
  }
  return(list(theta = theta, proposal.accepted = proposal.accepted))
}

sv.data = generate_data(0.91, 0.16, 0.64, 100)
num_iteration = 15000
burn.in = 5000
pmh.results = pmh(sv.data$y, 0.1, 0.1, 200, 100, num_iteration, c(0.01, 0.2), multinomial.resampling)

pmh.posterior.sigma2 = tibble(sigma2 = pmh.results$theta[1,burn.in:num_iteration])
pmh.posterior.beta2 = tibble(beta2 = pmh.results$theta[2,burn.in:num_iteration])

acceptance.ratio = sum(pmh.results$proposal.accepted)/length(pmh.results$proposal.accepted)
print(acceptance.ratio)
# 0.4444

# sigma^2
ggplot(pmh.posterior.sigma2, aes(x = sigma2)) +
  geom_histogram(bins = 30, fill="darkblue", color="darkblue", aes(y = ..density..))  +
  geom_density(alpha = 0.2, fill = "lightblue") +
  geom_vline(xintercept = mean(pmh.posterior.sigma2$sigma2), size = 1.2, linetype = 3, color = "black") +
  xlab(expression(sigma^{2})) +
  ylab("") +
  scale_y_discrete(breaks = NULL) + theme_bw() +
  annotate('text', x = 0.06, y = 30,
           label = paste("bar(sigma^2)==~",round(mean(pmh.posterior.sigma2$sigma2), 4)),
           parse = TRUE,size=10)

ggsave("plots/sigma2.png")

# beta^2
ggplot(pmh.posterior.beta2, aes(x = beta2)) +
  geom_histogram(bins = 30, fill="darkblue", color="darkblue", aes(y = ..density..))  +
  geom_density(alpha = 0.2, fill = "lightblue") +
  geom_vline(xintercept = mean(pmh.posterior.beta2$beta2), size = 1.2, linetype = 3, color = "black") +
  xlab(expression(beta^{2})) +
  ylab("") + theme_bw() + 
  annotate('text', x = 1, y =2,
           label = paste("bar(beta^2)==~",round(mean(pmh.posterior.beta2$beta2), 4)),
           parse = TRUE,size=10)  +
  scale_y_continuous(breaks = NULL)
ggsave("plots/beta2.png")


tibble(sigma2 = pmh.results$theta[2,]) %>% mutate(iterations = 1:num_iteration) %>% ggplot(aes(x = iterations,
                                                                                            y = sigma2)) + geom_line() + theme_bw()


