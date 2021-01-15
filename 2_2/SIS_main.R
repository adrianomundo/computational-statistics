library(tibble)
library(ggplot2)
library(magrittr)

source("SV_data_generator.R")
source("SIS.R")

set.seed(2723)
# data generation
gen.data = generate_data(0.91, 0.16, 0.64, 100)
sis.filter = sis(my.data$y, 1500, 100, 0.91, 0.16, 0.64)

# point estimate and MSE
print(gen.data$x[100])
print(sis.filter$x.hat[100])
print("MSE - SIS:" ) 
print(sis.filter$x.hat[100] - gen.data$x[100]^2)

# weights histogram
tibble(
  value = as.numeric(sis.filter$wnorm[100, ])) %>% 
  ggplot(aes(x = value)) +
  geom_histogram(fill="darkblue", color="darkblue", bins = 100) + theme_bw()
ggsave("plots/sis_weights_histogram.png")

# comparison
tibble(T = 1:100, Value = sis.filter$x.hat) %>% 
  ggplot(aes(x = T, y = Value)) +
  geom_line(aes(color = "a")) +
  geom_line(data = tibble(t = 1:100, x = gen.data$x), aes(x = t, y = x, color = "b")) +
  ggtitle("Sampled vs Test Value Comparison") +
  scale_color_manual(name = NULL, breaks = c("a", "b"), 
                     labels = c("SIS", "Test Data"), values = c("blue", "black")) + theme_bw()
  
# MSE
mse.tibble.sis = tibble(N = as.numeric(), mse = as.numeric())
for(N in seq(10, 1500, 5)){
  sis.filter = sis(gen.data$y, N, 100, 0.91, 0.16, 0.64)
  mse.tibble.sis = add_row(mse.tibble.sis, N = N, mse = (sis.filter$x.hat[100] - gen.data$x[100])^2)
}
ggplot(mse.tibble.sis, aes(x = N, y = mse)) + 
  geom_line() + ylab("MSE") + theme_bw()
ggsave("plots/sis_MSE.png")

# empirical variance
print(var(sis.filter$wnorm[100,]))

