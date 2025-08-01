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

multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

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

plt_frame <- data.frame(sapply(c(
  "blockforest",
  "priority_elastic_net",
  "cox_intermediate_concat",
  "salmon_salmon",
  "multimodal_nsclc",
  "survival_net_survival_net",
  "eh_intermediate_concat",
  "gdp_gdp",
  "customics_customics",
  "multimodal_survival_pred_multimodal_survival_pred",
  "cox_late_mean",
  "eh_late_mean"
),
       function(model_choice) {
         multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == model_choice) %>%
           group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>% pull(mean)
       }
       
))

plt_frame <- cbind(  project = multimodal_metrics %>% filter(modalities == "clinical_gex") %>% filter(model == "cox_late_mean") %>%
          group_by(project, cancer) %>% summarise(mean = mean(antolini_concordance)) %>%
          pull(project), plt_frame)
colnames(plt_frame) <-    c(
  "project",
  "BlockForest",
  "PriorityLasso L1+L2",
  "NN Cox IC",
  "Salmon",
  "Multimodal NSCLC",
  "SurvivalNet",
  "NN EH IC",
  "GDP",
  "CustOmics",
  "Mult. Surv. Prediction",
  "NN Cox LM",
  "NN EH LM"
)





a <- plt_frame %>% ggpairs(aes(color = project),
  lower = list(continuous = wrap("points", size = 3)),
  upper = list(continuous = "blank"),
  diag = list(continuous = "blankDiag"),
  legend = 1
) + ggpubfigs::theme_big_simple() + geom_abline(lty = 2)

final_pairs <- gpairs_lower(a) + labs(color = "") + scale_color_manual(values = ggpubfigs::friendly_pals$muted_nine) + labs(x = "Antolini's C", y = "Antolini's C")

ggsave("./figures_reproduced/survboard_final_fig_S3.pdf", width = 36, height = 15, plot = final_pairs)
ggsave("./figures_reproduced/survboard_final_fig_S3.svg", width = 36, height = 15, plot = final_pairs)
