library(vroom)
library(ggplot2)
library(ggpubfigs)
library(dplyr)
library(tidyr)
library(rcartocolor)

nColor <- 12
colors <- carto_pal(nColor, "Safe")
scales::show_col(colors)

set.seed(42)
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

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
    "eh_intermediate_concat",
    "cox_late_mean",
    "eh_late_mean",
    
    "salmon_salmon",
    "multimodal_nsclc",
    "survival_net_survival_net",
    
    "gdp_gdp",
    "customics_customics",
    "multimodal_survival_pred_multimodal_survival_pred"
    
    
  )
)

a_legend_left <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & !grepl("discrete_time", model)) %>%
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
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "crossbar", aes(color = "Mean"), width = 0.2, lty = 2, show.legend = FALSE) +
  stat_summary(fun.y = median, geom = "crossbar", aes(color = "Median"), width = 0.2, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (Antolini's C)", fill = "") +
  scale_fill_manual(values = colors) +
  # scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven) +
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
  guides(fill = guide_legend(override.aes = list(size = 10), nrow = 2)) +
  theme(
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.box = "horizontal"
  )


a <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & !grepl("discrete_time", model)) %>%
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
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (Antolini's C)", fill = "") +
  scale_fill_manual(values = colors) +
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




b <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & !grepl("discrete_time", model)) %>%
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
  summarise(rank = (mean(integrated_brier_score))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  theme_big_simple() +
  labs(x = "", y = "Rank (IBS)", fill = "") +
  scale_fill_manual(values = colors) +
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

c <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & !grepl("discrete_time", model)) %>%
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
  summarise(rank = (mean(d_calibration))) %>%
  ungroup() %>%
  group_by(cancer, project) %>%
  mutate(rank = rank(rank)) %>%
  ggplot(aes(x = model, y = rank, fill = model)) +
  geom_violin(draw_quantiles = c(0.5), bw = 0.55) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  theme_big_simple() +
  labs(x = "", y = "Rank (D-CAL)", fill = "") +
  scale_fill_manual(values = colors) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))


multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]
external_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_transfer.csv")[, -1]

multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

external_metrics <- external_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

multimodal_metrics <- multimodal_metrics %>%
  filter(cancer %in% c("LIRI-JP", "PAAD")) %>%
  filter((modalities == "clinical_gex") & !grepl("discrete_time", model))


external_metrics <- external_metrics %>% filter(!grepl("discrete_time", model))

external_metrics$cancer <- ifelse(
  external_metrics$cancer == "LIHC",
  "LIRI-JP",
  "PAAD"
)

internal <- multimodal_metrics %>% arrange(desc(cancer), desc(model))

external <- external_metrics %>% arrange(desc(cancer), desc(model))

plt_frame <- data.frame(
  model = internal$model,
  cancer = internal$cancer,
  internal = internal$antolini_concordance,
  external = external$antolini_concordance
)

plt_frame$model <- factor(plt_frame$model,   levels = c(
  "blockforest",
  "priority_elastic_net",
  
  "cox_intermediate_concat",
  "eh_intermediate_concat",
  "cox_late_mean",
  "eh_late_mean",
  
  "salmon_salmon",
  "multimodal_nsclc",
  "survival_net_survival_net",
  
  "gdp_gdp",
  "customics_customics",
  "multimodal_survival_pred_multimodal_survival_pred"
  
  
))

# plt_frame$cancer <- ifelse(
#  plt_frame$cancer == "PAAD",
#  "ICGC-PACA-CA --> TCGA-PAAD",
#  "TCGA-LIHC --> ICGC-LIRI-JP"
# )

fig_2_ext_a <- plt_frame %>%
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
  ggplot(aes(y = external, x = model, fill = model)) +
  geom_boxplot() +
  facet_wrap(~cancer, scales = "free") +
  theme_big_simple() +
  labs(x = "", y = "Antolini's C", fill = "") +
  scale_fill_manual(values = colors) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))

plt_frame <- data.frame(
  model = internal$model,
  cancer = internal$cancer,
  internal = internal$integrated_brier_score,
  external = external$integrated_brier_score
)

fig_2_ext_b <- plt_frame %>%
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
  ggplot(aes(y = external, x = model, fill = model)) +
  geom_boxplot() +
  facet_wrap(~cancer, scales = "free") +
  theme_big_simple() +
  labs(x = "", y = "IBS", fill = "") +
  scale_fill_manual(values = colors) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))

plt_frame <- data.frame(
  model = internal$model,
  cancer = internal$cancer,
  internal = internal$d_calibration,
  external = external$d_calibration
)

fig_2_ext_c <- plt_frame %>%
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
  ggplot(aes(y = external, x = model, fill = model)) +
  geom_boxplot() +
  facet_wrap(~cancer, scales = "free") +
  theme_big_simple() +
  labs(x = "", y = "D-CAL", fill = "") +
  scale_fill_manual(values = colors) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))

### Runtime etc
runtime <- c()
memory <- c()
rep <- c()
cancer_list <- c()
model_list <- c()
for (model in c(
  "blockforest",
  "prioritylasso",
  "late_mean_cox",
  "late_mean_eh",
  "intermediate_concat_cox",
  "intermediate_concat_eh",
  "salmon",
  "gdp",
  "survival_net",
  "customics",
  "denoising_ae",
  "multimodal_survival_pred"
)) {
  for (cancer in c(
    "BLCA",
    "BRCA",
    "COAD",
    "ESCA",
    "HNSC",
    "KIRC",
    "KIRP",
    "LGG",
    "LUAD",
    "PAAD",
    "SARC",
    "SKCM",
    "STAD",
    "UCEC",
    "OV",
    "LIHC",
    "LUSC",
    "LAML",
    "CESC",
    "GBM",
    "READ"
  )) {
    if (model == "salmon") {
      runtime <- c(
        runtime,
        vroom::vroom(
          paste0("./benchmarks/timings/", "make_salmon_data", "_", cancer, ".tsv")
        )$s +
          vroom::vroom(
            paste0("./benchmarks/timings/", model, "_", cancer, ".tsv")
          )$s
      )

      memory <- c(
        memory,
        apply(data.frame(
          X1 = vroom::vroom(
            paste0("./benchmarks/timings/", model, "_", cancer, ".tsv")
          )$max_uss,
          X2 = vroom::vroom(
            paste0("./benchmarks/timings/", "make_salmon_data", "_", cancer, ".tsv")
          )$max_uss
        ), 1, max)
      )
    } else {
      runtime <- c(
        runtime,
        vroom::vroom(
          paste0("./benchmarks/timings/", model, "_", cancer, ".tsv")
        )$s
      )
      memory <- c(
        memory,
        vroom::vroom(
          paste0("./benchmarks/timings/", model, "_", cancer, ".tsv")
        )$max_uss
      )
    }

    rep <- c(
      rep,
      1:5
    )
    cancer_list <- c(
      cancer_list,
      rep(cancer, 5)
    )
    model_list <- c(
      model_list,
      rep(model, 5)
    )
  }
}
### External validation

plt_frame <- data.frame(
  runtime = runtime,
  memory = memory,
  rep = rep,
  cancer = cancer_list,
  model = model_list
)

unique(plt_frame$model)

plt_frame %>%
  filter(!model %in%  c("prioritylasso", "blockforest")) %>%
  #filter(model == "multimodal_survival_pred") %>%
  #group_by(model) %>%
  summarise(mean = mean(memory))

#plt_frame %>%
#  filter(model %in% ) %>%
#  group_by(model) %>%
#  summarise(mean = mean(memory))
# plt_frame %>% group_by(model) %>% summarise(mean=mean(memory))

plt_frame$model <- factor(plt_frame$model, levels = c(
  "blockforest",
  "prioritylasso",
  
  "intermediate_concat_cox",
  "intermediate_concat_eh",
  "late_mean_cox",
  "late_mean_eh",
  "salmon",
  "denoising_ae",
  "survival_net",
  
  "gdp",
  "customics",
  "multimodal_survival_pred"

))

plt_frame <- plt_frame %>% arrange(desc(model), desc(cancer), desc(rep))

# plt_frame$runtime <- plt_frame$runtime / plt_frame[plt_frame$model == "prioritylasso", ]$runtime
# plt_frame$memory <- plt_frame$memory / plt_frame[plt_frame$model == "prioritylasso", ]$memory

plt_frame$cancer <- factor(plt_frame$cancer,
  levels = c(
    "BLCA",
    "BRCA",
    "COAD",
    "ESCA",
    "HNSC",
    "KIRC",
    "KIRP",
    "LGG",
    "LUAD",
    "PAAD",
    "SARC",
    "SKCM",
    "STAD",
    "UCEC",
    "OV",
    "LIHC",
    "LUSC",
    "LAML",
    "CESC",
    "GBM",
    "READ"
  )
)


runtime <- plt_frame %>%
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
    `denoising_ae` = "Multimodal NSCLC",
    `multimodal_survival_pred_multimodal_survival_pred` = "Multimodal Survival Prediction",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  group_by(model, cancer) %>%
  summarise(mean = mean(runtime), sd = sd(runtime)) %>%
  ungroup() %>%
  ggplot(aes(x = cancer, y = mean, color = model)) +
  geom_line(aes(group = model), size = 2) +
  theme_big_simple() +
  labs(x = "", y = "Mean runtime (seconds)", fill = "", color = "") +
  scale_color_manual(values = colors) +
  theme(axis.text.x = element_text(angle = 90)) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))


memory <- plt_frame %>%
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
    `denoising_ae` = "Multimodal NSCLC",
    `multimodal_survival_pred_multimodal_survival_pred` = "Multimodal Survival Prediction",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  group_by(model, cancer) %>%
  summarise(mean = mean(memory), sd = sd(memory)) %>%
  ungroup() %>%
  ggplot(aes(x = cancer, y = mean, color = model)) +
  geom_line(aes(group = model), size = 2) +
  theme_big_simple() +
  labs(x = "", y = "Mean memory (USS MB)", fill = "", color = "") +
  scale_color_manual(values = colors) +
  theme(axis.text.x = element_text(angle = 90)) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))


library(vroom)
library(ggplot2)
library(ggpubfigs)
library(dplyr)
library(tidyr)
set.seed(42)
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]
multimodal_metrics$model <- factor(
  multimodal_metrics$model,
  levels = c(
    "blockforest",
    "priority_elastic_net",
    
    "cox_intermediate_concat",
    "eh_intermediate_concat",
    "cox_late_mean",
    "eh_late_mean",
    
    "salmon_salmon",
    "multimodal_nsclc",
    "survival_net_survival_net",
    
    "gdp_gdp",
    "customics_customics",
    "multimodal_survival_pred_multimodal_survival_pred"
  )
)
failures <- multimodal_metrics %>%
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
  filter(modalities == "clinical_gex" & project == "TCGA") %>%
  group_by(model, cancer) %>%
  summarise(failures = sum(failures)) %>%
  ungroup() %>%
  ggplot(aes(
    x = cancer, y = failures, color = model
  )) +
  geom_line(aes(group = model), size = 2) +
  theme_big_simple() +
  labs(x = "", y = "Number of failures", fill = "", color = "") +
  scale_color_manual(values = colors) +
  theme(axis.text.x = element_text(angle = 90)) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))




fig_2 <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "no"), ncol = 3, labels = c("A", "B", "C"), label_size = 24)

fig_2_second <- cowplot::plot_grid(fig_2_ext_a + theme(legend.position = "none"), fig_2_ext_b + theme(legend.position = "none"), fig_2_ext_c + theme(legend.position = "none"), ncol = 3, labels = c("D", "E", "F"), label_size = 24)


fig_3_third <- cowplot::plot_grid(cowplot::plot_grid(runtime + theme(legend.position = "none"), memory + theme(legend.position = "none"), failures + theme(legend.position = "none"), ncol = 3, labels = c("G", "H", "I"), label_size = 24),
  cowplot::ggdraw(cowplot::get_plot_component(a_legend_left, "guide-box-bottom", return_all = TRUE)),
  nrow = 2,
  rel_heights = c(0.9, 0.25)
)

final_fig <- cowplot::plot_grid(
  fig_2,
  fig_2_second,
  fig_3_third,
  nrow = 3,
  rel_heights = c(1 / 2, 1 / 2, 1)
)


ggsave("./figures_reproduced/survboard_final_fig_3.pdf", final_fig, width = 26, height = 16)
ggsave("./figures_reproduced/survboard_final_fig_3.svg", final_fig, width = 26, height = 16)
