library(tibble)
library(dplyr)
library(ggplot2)
library(magrittr)
library(purrr)

source("../2_2/BPF.R")
source("../2_2/SV_data_generator.R")
source("loglikelihood.R")

set.seed(2723)
sv.data = generate_data(0.91, 0.16, 0.64, 100)

# T Variation bpf multinomial - N = 80
variation.T = tibble(T = numeric(), logl = numeric())
for(t in c(25, 40, 50, 60, 75, 90, 100)){
  # n run for each T 
  for(i in 1:100){ 
    variation.T = add_row(variation.T, T = t, logl = bpf(sv.data$y, 0.91, 0.16, 0.64, 80, t, multinomial.resampling)$logl)
  }
}
ggplot(variation.T, aes(x = T, group = T, y = logl)) +
  geom_boxplot(color="red", fill="orange") + ylab("log-likelihood") + xlab("T") + theme_bw()
ggsave("plots/variation_T_boxplot.png")

# N Variation bpf multinomial - T = 100
variation.N = tibble(N = numeric(), logl = numeric())
for(n in c(25, 50, 100, 200, 300, 500)){
  # n run for each N 
  for(i in 1:100){ 
    variation.N = add_row(variation.N, N = n, logl = bpf(sv.data$y, 0.91, 0.16,0.64, n, 100, multinomial.resampling)$logl)
  }
}
ggplot(variation.N, aes(x = N, group = N, y = logl)) +
  geom_boxplot(color="red", fill="orange") + ylab("log-likelihood") + xlab("N")
ggsave("plots/variation_N_boxplot.png")
