library(tibble)
library(ggplot2)
library(magrittr)

# Synthetic stochastic-volatility data generation
generate_data = function(phi, sigma, beta, data_points) {
  
  sv_data = tibble(x = numeric(), y = numeric(), t = numeric())
  x = rnorm(1, 0, sigma)
  y = rnorm(1, 0, sqrt(beta^2 * exp(x)))
  sv_data = tibble::add_row(sv_data, x = x, y = y, t = 1)
  
  for(i in 2:data_points) {
    x = rnorm(1, phi*x, sigma)
    y = rnorm(1, 0, sqrt(beta^2 * exp(x)))
    sv_data = tibble::add_row(sv_data, x = x, y = y, t = i)
  }
  
  return(sv_data)
}

# plot comparison with sv data
tibble(T = 1:100, Value = sv.data$x) %>% 
  ggplot(aes(x = T, y = Value)) +
  #geom_line(aes(color = "a")) +
  geom_line(data = tibble(t = 1:100, x = sv.data$x), aes(x = t, y = x, color = "black"), alpha = 0.4, color="darkblue") + theme_bw()
ggsave("2_2/plots/sv_data.png")