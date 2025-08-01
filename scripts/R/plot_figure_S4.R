library(vroom)
library(ggplot2)
library(ggpubfigs)
library(dplyr)
library(tidyr)
set.seed(42)
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

library(rcartocolor)

nColor <- 12
colors <- carto_pal(nColor, "Safe")
scales::show_col(colors)

multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

multimodal_metrics$model <- factor(
  multimodal_metrics$model,
  levels = c(
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
  )
)

a_legend_left <- multimodal_metrics %>% 
  filter(modalities == "clinical_gex") %>%
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
  scale_fill_manual(values = colors) +
  facet_wrap(~project)+ 
  #scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven) +
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
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.box = "horizontal"
  )

ggsave("./figures_reproduced/survboard_final_fig_S4.pdf", plot = a_legend_left, width = 18, height = 8)
ggsave("./figures_reproduced/survboard_final_fig_S4.svg", plot = a_legend_left, width = 18, height = 8)
