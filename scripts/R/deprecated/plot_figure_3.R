library(vroom)
library(ggplot2)
library(ggpubfigs)
library(dplyr)
library(tidyr)

multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

a_legend_left <- multimodal_metrics %>% filter(modalities == "clinical_gex") %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical + GEX)",
    `eh_intermediate_concat` = "NN EH IC (Clinical + GEX)",
    `eh_late_mean` = "NN EH LM (Clinical + GEX)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical + GEX)"
  )) %>%
  mutate(modalities = recode(
    modalities,
    `clinical` = "Clinical",
    `clinical_gex` = "Clinical + GEX",
    `cnv` = "CNV",
    `full` = "All modalities",
    `gex` = "GEX",
    `meth` = "DNA methylation",
    `mirna` = "miRNA",
    `mut` = "Mutation",
    `rppa` = "RPPA"
  )) %>%
  group_by(cancer, project, model, modalities) %>%
  summarise(rank = (mean(antolini_concordance))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(-rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw=0.55) +
  stat_summary(fun.y = mean, geom = "crossbar", aes(color = "Mean"), width = 0.2, lty = 2, show.legend = FALSE) +
  stat_summary(fun.y = median, geom = "crossbar", aes(color = "Median"), width = 0.2, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (Antolini's C)", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  scale_colour_manual(
    values = c("red", "blue"),
    name = ""
  ) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.box = "horizontal"
  )


a <- multimodal_metrics %>% filter(modalities == "clinical_gex") %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical + GEX)",
    `eh_intermediate_concat` = "NN EH IC (Clinical + GEX)",
    `eh_late_mean` = "NN EH LM (Clinical + GEX)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical + GEX)"
  )) %>%
  mutate(modalities = recode(
    modalities,
    `clinical` = "Clinical",
    `clinical_gex` = "Clinical + GEX",
    `cnv` = "CNV",
    `full` = "All modalities",
    `gex` = "GEX",
    `meth` = "DNA methylation",
    `mirna` = "miRNA",
    `mut` = "Mutation",
    `rppa` = "RPPA"
  )) %>%
  group_by(cancer, project, model, modalities) %>%
  summarise(rank = (mean(antolini_concordance))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(-rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw=0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (Antolini's C)", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))




b <- multimodal_metrics %>% filter(modalities == "clinical_gex" ) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox (Intermediate concat) (Clinical + GEX)",
    `cox_late_mean` = "NN Cox (Late mean) (Clinical + GEX)",
    `eh_intermediate_concat` = "NN EH (Intermediate concat) (Clinical + GEX)",
    `eh_late_mean` = "NN EH (Late mean) (Clinical + GEX)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "Priority elastic net (Clinical + GEX)"
  )) %>%
  mutate(modalities = recode(
    modalities,
    `clinical` = "Clinical",
    `clinical_gex` = "Clinical + GEX",
    `cnv` = "CNV",
    `full` = "All modalities",
    `gex` = "GEX",
    `meth` = "DNA methylation",
    `mirna` = "miRNA",
    `mut` = "Mutation",
    `rppa` = "RPPA"
  )) %>%
  group_by(cancer, project, model, modalities) %>%
  summarise(rank = (mean(integrated_brier_score))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw=0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (IBS)", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  guides(fill = guide_legend(override.aes = list(size = 10)))

c <- multimodal_metrics %>% filter(modalities == "clinical_gex") %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical + GEX)",
    `eh_intermediate_concat` = "NN EH IC (Clinical + GEX)",
    `eh_late_mean` = "NN EH LM (Clinical + GEX)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical + GEX)"
  )) %>%
  mutate(modalities = recode(
    modalities,
    `clinical` = "Clinical",
    `clinical_gex` = "Clinical + GEX",
    `cnv` = "CNV",
    `full` = "All modalities",
    `gex` = "GEX",
    `meth` = "DNA methylation",
    `mirna` = "miRNA",
    `mut` = "Mutation",
    `rppa` = "RPPA"
  )) %>%
  group_by(cancer, project, model, modalities) %>%
  summarise(rank = (mean(d_calibration))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw=0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  theme_big_simple() +
  labs(x = "", y = "Rank (D-CAL)", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))

fig_2 <- cowplot::plot_grid(cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "no"), ncol = 3, labels = c("A", "B", "C"), label_size = 24),
  cowplot::get_legend(a_legend_left),
  nrow = 2,
  rel_heights = c(0.9, 0.25)
)

ggsave("./figures_reproduced/survboard_final_fig_3.pdf", fig_2, width = 18, height = 5)
ggsave("./figures_reproduced/survboard_final_fig_3.svg", fig_2, width = 18, height = 5)
