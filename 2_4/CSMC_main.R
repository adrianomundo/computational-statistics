library(tibble)
library(dplyr)
library(ggplot2)
library(magrittr)
library(purrr)

source("../2_2/SV_data_generator.R")
source("../2_2/BPF.R")
source("CSMC.R")

set.seed(23423432)
sv.data = generate_data(0.91, 0.16, 0.64, 100)

set.seed(2723)
num_iterations = 20000
burn_in = 5000

results = pg(sv.data$y, 0.1, 0.1, 100, 100, num_iterations)

plot(results$sigma.density)
plot(results$beta.density)

pg.posterior.sigma_square = tibble(sigma_square = results$sigma.density[burn_in:20000])

# plot sigma sqaure with pg
ggplot(pg.posterior.sigma_square, aes(x = sigma_square)) +
  geom_histogram(bins = 30, fill="darkblue", color="darkblue", aes(y = ..density..))  +
  geom_density(alpha = 0.2, fill = "lightblue") +
  geom_vline(xintercept = mean(pg.posterior.sigma_square$sigma_square), size = 1.2, linetype = 3, color = "black") +
  xlab(expression(sigma^{2})) + ylab("") +
  scale_y_discrete(breaks = NULL) +
  annotate('text', x = 0.06, y = 30,
           label = paste("bar(sigma^2)==~",round(mean(pg.posterior.sigma_square$sigma_square), 4)),
           parse = TRUE,size=10) + theme_bw()
ggsave("plots/pg_sigma_square.png")

pg.posterior.beta_square = tibble(beta_square = results$beta.density[burn_in:20000])

# plot beta square with pg
ggplot(pg.posterior.beta_square, aes(x = beta_square)) +
  geom_histogram(bins = 30, fill="darkblue", color="darkblue", aes(y = ..density..))  +
  geom_density(alpha = 0.2, fill = "lightblue") +
  geom_vline(xintercept = mean(pg.posterior.beta_square$beta_square), size = 1.2, linetype = 3, color = "black") +
  xlab(expression(beta^{2})) + ylab("") +
  annotate('text', x = 0.85, y = 2,
           label = paste("bar(beta^2)==~",round(mean(pg.posterior.beta_square$beta_square), 4)),
           parse = TRUE,size=10) +
  scale_y_continuous(breaks = NULL) + theme_bw()
ggsave("plots/pg_beta_square.png")

tibble(logl = results$logl, k = 1:num_iterations) %>% ggplot(aes(k, logl)) + geom_line()
