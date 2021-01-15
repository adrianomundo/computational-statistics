library(tibble)
library(ggplot2)
library(magrittr)
library(RColorBrewer)

source("BPF.R")
source("SIS.R")
source("SV_data_generator.R")

set.seed(2723)
sv.data <- generate_data(0.91, 0.16, 0.64, 100)

# Compare all three methods
sis.filter <- sis(sv.data$y, 1500, 100, 0.91, 0.16, 0.64)

ggplot(data = tibble(T = 1:100, Value = bpf.multinomial$x.hat), aes(x = T, y = Value, color = "c")) +
  geom_line(aes(color = "b")) +
  geom_line(data = tibble(t = 1:100, x = sv.data$x), aes(x = t, y = x, color = "a") , linetype = "dashed") +
  geom_line(data = tibble(t = 1:100, x = sis.filter$x.hat), aes(t, x, color = "b")) +
  geom_line(data = tibble(t = 1:100, x = bpf.stratified$x.hat), aes(t, x)) +
  geom_line(aes(color = "d")) +
  scale_color_manual(name = NULL,
                     breaks = c("a", "b", "c", "d"),
                     labels = c("Test Data", "SIS", "BPF Multinomial", "BPF Stratified"),
                     values = c("#666666", "#D95F02", "#66A61E", "#7570B3")) +
  theme(legend.text = element_text(size = 12)) + theme_bw()
ggsave("plots/pf_comparison.png", units = c("cm"), width = 40)

