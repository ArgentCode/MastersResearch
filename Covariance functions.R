library(ggplot2)
library(dplyr)

k <- seq(0.5, 75, by = 0.5)   # avoid k=0, cap at 100 (1000 is overkill visually)
r1 <- 0.4
r2 <- 0.97
d <- 0.3
alpha <- 0.4
c <- 20

df <- data.frame(
  k = rep(k, 3),
  gamma = c(
    c * r1^k, # short
    c * k^(alpha - 1) * r2^k, # semi-long
    c * k^(2 * d - 1) # long
  ),
  type = rep(c("Short Memory", "Semi-Long Memory", "Long Memory"), each = length(k))
)

df$type <- factor(df$type, levels = c("Short Memory", "Semi-Long Memory", "Long Memory"))

ggplot(df, aes(x = k, y = gamma, color = type)) +
  geom_line(linewidth = 0.9) +
  scale_color_manual(
    values = c(
      "Short Memory"     = "#E64B35",
      "Semi-Long Memory" = "#F5A623",
      "Long Memory"      = "#4A90D9"
    )
  ) +
  scale_y_continuous(limits = c(0, 30)) +
  labs(
    title    = "Autocovariance Function by Memory Type",
    x        = "Lag (k)",
    y        = expression(gamma(k)),
    color    = NULL
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title      = element_text(face = "bold", size = 16),
    plot.subtitle   = element_text(size = 11, color = "grey40", margin = margin(b = 10)),
    legend.position = "bottom",
    legend.text     = element_text(size = 12),
    panel.grid.minor = element_blank(),
    plot.margin     = margin(15, 15, 15, 15)
  )