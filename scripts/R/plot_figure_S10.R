library(ggpubfigs)
library(dplyr)
library(caret)
library(survival)
library(survminer)
library(survcomp)
library(tidyr)
library(pheatmap)
library(RColorBrewer)
library(GGally)
library(tidyr)


unimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% mutate(model = recode(model, `eh_early` = "eh_intermediate_concat", `cox_early` = "cox_intermediate_concat", `elastic_net` = "priority_elastic_net", `rsf` = "blockforest"))
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

unimodal_metrics <- unimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()
multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

# Adapted from: https://stackoverflow.com/questions/42654928/how-to-show-only-the-lower-triangle-in-ggpairs
gpairs_lower <- function(g) {
  g$plots <- g$plots[-c((1:g$nrow), (g$nrow:(2 * g$nrow)))]
  g$yAxisLabels <- g$yAxisLabels[-(1:2)]
  g$nrow <- g$nrow - 2

  g$plots <- g$plots[-(seq(1, length(g$plots), by = g$ncol))]
  g$plots <- g$plots[-(seq(g$ncol - 1, length(g$plots) / g$ncol * (g$ncol - 1), by = g$ncol - 1))]
  g$xAxisLabels <- g$xAxisLabels[-1]
  g$xAxisLabels <- g$xAxisLabels[-length(g$xAxisLabels)]
  g$ncol <- g$ncol - 2

  g
}

a <- data.frame(
  project = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "cox_late_mean") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(project),
  `NN Cox IC (Clinical + GEX)` = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "cox_intermediate_concat") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>% pull(mean),
  `NN Cox LM (Clinical + GEX)` = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "cox_late_mean") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(mean),
  `NN EH IC (Clinical + GEX)` = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "eh_intermediate_concat") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(mean),
  `NN EH LM (Clinical + GEX)` = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "eh_late_mean") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(mean),
  `BlockForest (Clinical + GEX)` = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "blockforest") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(mean),
  `PriorityLasso L1+L2 (All modalities)` = multimodal_metrics %>% filter(modalities == "full") %>% filter(model == "priority_elastic_net") %>%
    group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
    pull(mean),
  check.names = FALSE
) %>% ggpairs(aes(color = project),
  lower = list(continuous = wrap("points", size = 3)),
  upper = list(continuous = "blank"),
  diag = list(continuous = "blankDiag"),
  legend = 1
) + ggpubfigs::theme_big_simple() + geom_abline(lty = 2)

final_pairs <- gpairs_lower(a) + labs(color = "") + scale_color_manual(values = ggpubfigs::friendly_pals$muted_nine) + labs(x = "Antolini's C", y = "Antolini's C")

ggsave("./figures_reproduced/survboard_final_fig_S10.pdf", width = 18 * 1.35, height = 5 * 1.5, plot = final_pairs)
ggsave("./figures_reproduced/survboard_final_fig_S10.svg", width = 18 * 1.35, height = 5 * 1.5, plot = final_pairs)
