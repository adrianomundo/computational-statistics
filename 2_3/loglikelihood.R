library(tibble)
library(dplyr)
library(ggplot2)
library(magrittr)

source("../2_2/BPF.R")
source("../2_2/SV_data_generator.R")

parameters.grid = tibble(sigma = numeric(), beta = numeric(), logl = numeric())

# 25 x 25 grid = 625 values 
for(s in seq(0.001, 2, by = 0.08)) {
  for(b in seq(0.001, 2, by = 0.08)) {
    parameters.grid = add_row(parameters.grid, sigma = s, beta = b)
  }
}

# data generation
sv.data = generate_data(0.91, 0.16, 0.64, 100)

for(row in 1:nrow(parameters.grid)){
  parameters.grid[row,"logl"] = loglikelihood(y = sv.data$y, phi = 0.91, 
                                           sigma = parameters.grid[row,"sigma"] %>% pull,
                                           beta = parameters.grid[row,"beta"] %>% pull,
                                           N = 200, T = 100, multinomial.resampling, times = 10)
}

# extract best parameters
parameters.grid %>% arrange(-logl) %>% top_n(20)

# Log-likelihood function calculation for each bpf 
loglikelihood = function(y, phi, sigma, beta, N, T, resampling.function, times) {
  results = c()
  for(i in 1:times) {
    results = c(results, bpf(y, phi, sigma, beta, N, T, resampling.function)$logl)
  }
  return(mean(results))
}

# mlogl = parameters.grid %>%  mutate(logl = ifelse(logl > -250, logl, NA)) %>% filter(!is.na(logl)) %>%
#   summarize(medianlogl = mean(logl)) %>% pull
# 
# parameters.grid %>% mutate(logl = ifelse(logl > -100, logl, NA)) %>%
#   ggplot(aes(x = sigma, y = beta, fill = logl)) +
#   geom_tile() + scale_fill_gradient2(midpoint = mlogl, low = "blue", mid = "white",
#                                      high = "red", space = "Lab") +
#   xlab(expression(sigma)) +
#   ylab(expression(beta))  +
#   theme(axis.title.y = element_text(angle = 0))



