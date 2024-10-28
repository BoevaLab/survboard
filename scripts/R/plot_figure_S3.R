library(ggpubfigs)
library(dplyr)
library(caret)
library(survival)
library(survminer)
library(survcomp)
library(tidyr)
library(pheatmap)
library(RColorBrewer)


unimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% mutate(model = recode(model, `eh_early` = "eh_intermediate_concat", `cox_early` = "cox_intermediate_concat", `elastic_net` = "priority_elastic_net", `rsf` = "blockforest"))
unimodal_metrics <- rbind(unimodal_metrics, vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% filter(!model %in% c("rsf", "elastic_net")) %>% mutate(model = recode(model, `eh_early` = "eh_late_mean", `cox_early` = "cox_late_mean")))
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

unimodal_metrics <- unimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()
multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()


a <- rbind(
  unimodal_metrics %>% filter(modalities == "clinical" & !model %in% c("blockforest", "cox_intermediate_concat")),
  multimodal_metrics %>% filter(modalities == "clinical_gex" & model %in% c("blockforest", "cox_intermediate_concat"))
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical)",
    `eh_intermediate_concat` = "NN EH IC (Clinical)",
    `eh_late_mean` = "NN EH LM (Clinical)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical)"
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
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (Antolini's C)", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  )

b <- rbind(
  unimodal_metrics %>% filter(modalities == "clinical" & !model %in% c("blockforest", "cox_intermediate_concat")),
  multimodal_metrics %>% filter(modalities == "clinical_gex" & model %in% c("blockforest", "cox_intermediate_concat"))
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical)",
    `eh_intermediate_concat` = "NN EH IC (Clinical)",
    `eh_late_mean` = "NN EH LM (Clinical)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical)"
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
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
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
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  )

c <- rbind(
  unimodal_metrics %>% filter(modalities == "clinical" & !model %in% c("blockforest", "cox_intermediate_concat")),
  multimodal_metrics %>% filter(modalities == "clinical_gex" & model %in% c("blockforest", "cox_intermediate_concat"))
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC (Clinical + GEX)",
    `cox_late_mean` = "NN Cox LM (Clinical)",
    `eh_intermediate_concat` = "NN EH IC (Clinical)",
    `eh_late_mean` = "NN EH LM (Clinical)",
    `blockforest` = "BlockForest (Clinical + GEX)",
    `priority_elastic_net` = "PriorityLasso L1+L2 (Clinical)"
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
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
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
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  )

cowplot::plot_grid(cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "no"), ncol = 3, labels = c("A", "B", "C"), label_size = 24),
  cowplot::get_legend(a),
  nrow = 2,
  rel_heights = c(0.9, 0.2)
)

ggsave("./figures_reproduced/survboard_final_fig_S3.pdf", width = 18, height = 5)
ggsave("./figures_reproduced/survboard_final_fig_S3.svg", width = 18, height = 5)
