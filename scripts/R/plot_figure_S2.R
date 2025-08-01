renv::status()
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
unimodal_metrics <- rbind(unimodal_metrics, vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% filter(!model %in% c("rsf", "elastic_net")) %>% mutate(model = recode(model, `eh_early`= "eh_late_mean", `cox_early` = "cox_late_mean")))
unimodal_metrics <- rbind(unimodal_metrics, vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% filter(!model %in% c("rsf", "elastic_net")) %>% mutate(model = recode(model, `eh_early`= "eh_intermediate_attention", `cox_early` = "cox_intermediate_attention")))
unimodal_metrics <- rbind(unimodal_metrics, vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% filter(!model %in% c("rsf", "elastic_net")) %>% mutate(model = recode(model, `eh_early`= "eh_early", `cox_early` = "cox_early")))
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

unimodal_metrics <- unimodal_metrics %>%
  filter(!grepl("(discrete_time|early|attention)", model)) %>%
  group_by(project, cancer, model, modalities) %>%
  
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()
multimodal_metrics <- multimodal_metrics %>%
  #filter(!grepl("discrete_time", model)) %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

plt_frame <- rbind(unimodal_metrics, multimodal_metrics[, -ncol(multimodal_metrics)])

plt_frame$model <- factor(plt_frame$model, levels = unique(multimodal_metrics$model))

b <- plt_frame %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC",
    `cox_late_mean` = "NN Cox LM",
    # `cox_intermediate_attention` = "NN Cox IA",
    # `cox_early` = "NN Cox EGL",
    `eh_intermediate_concat` = "NN EH IC",
    `eh_late_mean` = "NN EH LM",
    # `eh_intermediate_attention` = "NN EH IA",
    # `eh_early` = "NN EH EGL",
    
    # `discrete_time_intermediate_concat` = "NN DT IC",
    # `discrete_time_late_mean` = "NN DT LM",
    # `discrete_time_intermediate_attention` = "NN DT IA",
    # `discrete_time_early` = "NN DT EGL",
    `salmon_salmon` = "Salmon",
    `gdp_gdp` = "GDP",
    `survival_net_survival_net` = "SurvivalNet",
    `customics_customics` = "CustOmics",
    `multimodal_nsclc` = "Multimodal NSCLC",
    `multimodal_survival_pred_multimodal_survival_pred` = "Multimodal Survival Prediction",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  mutate(modalities = recode(
    modalities,
    `clinical` = "Clinical (n=28)",
    `clinical_gex` = "Clinical + GEX (n=28)",
    `cnv` = "CNV (n=23)",
    `full` = "All modalities (n=28)",
    `gex` = "GEX (n=28)",
    `meth` = "DNA methylation (n=22)",
    `mirna` = "miRNA (n=21)",
    `mut` = "Mutation (n=25)",
    `rppa` = "RPPA (n=17)"
  ))

b <- b %>% group_by(cancer, project, model, modalities) %>%
  summarise(rank = (mean(integrated_brier_score))) %>%
  ungroup() %>%
  group_by(cancer, project, model) %>%
  mutate(rank = rank(rank)) %>%
  ggplot(aes(x = modalities, y = rank, fill = modalities)) +
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (IBS)", fill = "") +
  scale_fill_manual(values = c(friendly_pals$contrast_three, friendly_pals$bright_seven[1:6])) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  facet_wrap(~model, scales = "free_y") +
  theme(legend.text = element_text(size = 20)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  )

ggsave("./figures_reproduced/survboard_final_fig_S2.pdf", width = 20, height = 8, plot = b)
ggsave("./figures_reproduced/survboard_final_fig_S2.svg", width = 20, height = 8, plot = b)
