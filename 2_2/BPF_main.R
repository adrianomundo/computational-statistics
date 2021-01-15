library(tibble)
library(ggplot2)
library(magrittr)
library(RColorBrewer)

source("BPF.R")
source("SV_data_generator.R")

# data generation
set.seed(2723)
sv.data = generate_data(0.91, 0.16, 0.64, 100)

# multinomial particle filter 
bpf.multinomial = bpf(sv.data$y, 0.91, 0.16, 0.64, 200, 100, multinomial.resampling)

# plot comparison with sv data
tibble(T = 1:100, value = bpf.multinomial$x.hat) %>%
  ggplot(aes(x = T, y = value)) +
  geom_line(aes(color = "a")) +
  geom_line(data = tibble(t = 1:100, x = sv.data$x), aes(x = t, y = x, color = "b"),
            alpha = 0.4) +
  ggtitle("Sampled vs Test Value Comparisons") +
  scale_color_manual(name = NULL, breaks = c("a", "b"),
                     labels = c("BPF Multinomial", "Test Data"), values = c("blue", "black")) + theme_bw()

# weights histogram
tibble(value = as.numeric(bpf.multinomial$w.norm[100, ])) %>% ggplot(aes(x = value)) +
  geom_histogram(fill="darkblue", color="darkblue", bins = 30) + theme_bw()
ggsave("plots/bpf_weights_multi_histogram.png")

# point estimate
bpf.multinomial$x.hat[100]
sv.data$x[100]
print("MSE - BPF Multi:" )
print((bpf.multinomial$x.hat[100] - sv.data$x[100])^2)

# MSE
mse.tibble.bpf.multinomial = tibble(N = as.numeric(), mse = as.numeric())
for(N in seq(10,1500, 5)){
  bpf.multinomial = bpf(sv.data$y, 0.91, 0.16, 0.64, N, 100, multinomial.resampling)
  mse.tibble.bpf.multinomial = add_row(mse.tibble.bpf.multinomial, N = N, mse = (bpf.multinomial$x.hat[100] - sv.data$x[100])^2)
}
ggplot(mse.tibble.bpf.multinomial, aes(x = N, y = mse)) +
  geom_line() + ylab("MSE") + theme_bw()
ggsave("plots/bpf_multinomial_MSE.png")

# empirical variance
print(var(bpf.multinomial$w.norm.bpf[100,]))

#-------------------------------------------------------------------------------------------------------------------

# stratified particle filter
bpf.stratified = bpf(sv.data$y, 0.91, 0.16, 0.64, 200, 100, stratified.resampling)

# plot comparison with sv data
tibble(T = 1:100, value = bpf.stratified$x.hat) %>%
  ggplot(aes(x = T, y = value)) +
  geom_line(aes(color = "a")) +
  geom_line(data = tibble(t = 1:100, x = sv.data$x), aes(x = t, y = x, color = "b"),
            alpha = 0.4) +
  ggtitle("Sampled vs Test Value Comparison") +
  scale_color_manual(name = NULL, breaks = c("a", "b"),
                     labels = c("BPF Stratified", "Test Data"), values = c("blue", "black")) + theme_bw()

# point estimate
bpf.stratified$x.hat[100]
sv.data$x[100]
print("MSE - BPF Strati:" )
print((bpf.stratified$x.hat[100] - sv.data$x[100])^2)

# weights degeneracy histogram
tibble(value = as.numeric(bpf.stratified$w.norm[100, ])) %>% ggplot(aes(x = value)) +
  geom_histogram(fill="darkblue", color="darkblue", bins = 30)  + theme_bw()
ggsave("plots/bpf_weights_stratified_histogram.png")

# MSE
mse.tibble.bpf.stratified = tibble(N = as.numeric(), mse = as.numeric())
for(N in seq(10,1500, 5)){
  bpf.stratified = bpf(sv.data$y, 0.91, 0.16, 0.64, N, 100, stratified.resampling)
  mse.tibble.bpf.stratified = add_row(mse.tibble.bpf.stratified, N = N, mse = (bpf.stratified$x.hat[100] - sv.data$x[100])^2)
}
ggplot(mse.tibble.bpf.stratified, aes(x = N, y = mse)) +
  geom_line() + ylab("MSE") + theme_bw()
ggsave("plots/bpf_stratified_MSE.png")

print(var(bpf.stratified$w.norm.bpf[100,]))

