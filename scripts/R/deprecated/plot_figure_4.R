library(vroom)
library(ggplot2)
library(ggpubfigs)
library(dplyr)
library(tidyr)


multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]
multimodal_metrics_missing <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal_missing.csv")[, -1]
pancancer_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_pancancer.csv")[, -1]

multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

multimodal_metrics_missing <- multimodal_metrics_missing %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

pancancer_metrics <- pancancer_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup()

diff_pancancer <- pancancer_metrics %>%
  filter(model %in% c("blockforest", "priority_elastic_net", "eh_late_mean", "eh_intermediate_concat", "cox_late_mean", "cox_intermediate_concat")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(antolini_concordance = mean(antolini_concordance)) %>%
  pull(antolini_concordance) - multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(antolini_concordance = mean(antolini_concordance)) %>%
  pull(antolini_concordance)

cancer_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)



a <- data.frame(
  diff = diff_pancancer,
  cancer = cancer_pancancer,
  model = model_pancancer
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC",
    `cox_late_mean` = "NN Cox LM",
    `eh_intermediate_concat` = "NN EH IC",
    `eh_late_mean` = "NN EH LM",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "Antolini's C", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  geom_hline(yintercept = 0, lty = 2) +
  annotate("text", x = 2, y = 0.15, label = "**", size = 8) +
  annotate("text", x = 3, y = 0.15, label = "**", size = 8) +
  annotate("text", x = 4, y = 0.15, label = "**", size = 8) +
  annotate("text", x = 5, y = 0.15, label = "**", size = 8) +
  annotate("text", x = 6, y = 0.15, label = "**", size = 8) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  labs(title = "Improvement from training on pan-cancer data") +
  guides(color = guide_legend(nrow = 2))

diff_pancancer <- -(pancancer_metrics %>%
  filter(model %in% c("blockforest", "priority_elastic_net", "eh_late_mean", "eh_intermediate_concat", "cox_late_mean", "cox_intermediate_concat")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(integrated_brier_score)) %>% pull(mean) - multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(integrated_brier_score)) %>% pull(mean))

cancer_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)

b <- data.frame(
  diff = diff_pancancer,
  cancer = cancer_pancancer,
  model = model_pancancer
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox (Intermediate concat)",
    `cox_late_mean` = "NN Cox (Late mean)",
    `eh_intermediate_concat` = "NN EH (Intermediate concat)",
    `eh_late_mean` = "NN EH (Late mean)",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "Priority elastic net"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "IBS", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  geom_hline(yintercept = 0, lty = 2) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  annotate("text", x = 2, y = 0.065, label = "**", size = 8) +
  annotate("text", x = 3, y = 0.065, label = "**", size = 8) +
  annotate("text", x = 4, y = 0.065, label = "**", size = 8) +
  annotate("text", x = 5, y = 0.065, label = "**", size = 8) +
  annotate("text", x = 6, y = 0.065, label = "*", size = 8) +
  labs(title = "")

diff_pancancer <- -(pancancer_metrics %>%
  filter(model %in% c("blockforest", "priority_elastic_net", "eh_late_mean", "eh_intermediate_concat", "cox_late_mean", "cox_intermediate_concat")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(d_calibration)) %>% pull(mean) - multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(d_calibration)) %>% pull(mean))

cancer_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_pancancer <- multimodal_metrics %>%
  filter(modalities == "clinical_gex") %>%
  filter(project == "TCGA") %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)

c <- data.frame(
  diff = diff_pancancer,
  cancer = cancer_pancancer,
  model = model_pancancer
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC",
    `cox_late_mean` = "NN Cox LM",
    `eh_intermediate_concat` = "NN EH IC",
    `eh_late_mean` = "NN EH LM",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "D-CAL", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  geom_hline(yintercept = 0, lty = 2) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  annotate("text", x = 1, y = 3, label = "*", size = 8) +
  annotate("text", x = 5, y = 3, label = "**", size = 8) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  labs(title = "")

legend_saved <- cowplot::get_legend(a)

figure_4_upper <- cowplot::plot_grid(cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), ncol = 3, labels = c("")))

diff_missing <- multimodal_metrics_missing %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(antolini_concordance = mean(antolini_concordance)) %>%
  pull(antolini_concordance) - multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(antolini_concordance = mean(antolini_concordance)) %>%
  pull(antolini_concordance)

cancer_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)

a <- data.frame(
  diff = diff_missing,
  cancer = cancer_missing,
  model = model_missing
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC",
    `cox_late_mean` = "NN Cox LM",
    `eh_intermediate_concat` = "NN EH IC",
    `eh_late_mean` = "NN EH LM",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "Antolini's C", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[2:7]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  geom_hline(yintercept = 0, lty = 2) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  annotate("text", x = 1, y = 0.125, label = "**", size = 8) +
  annotate("text", x = 2, y = 0.125, label = "**", size = 8) +
  annotate("text", x = 3, y = 0.125, label = "**", size = 8) +
  annotate("text", x = 4, y = 0.125, label = "**", size = 8) +
  labs(title = "Improvement from using samples with missing modalities as additional training data")

diff_missing <- -(multimodal_metrics_missing %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(integrated_brier_score)) %>% pull(mean) - multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(integrated_brier_score)) %>% pull(mean))

cancer_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)

b <- data.frame(
  diff = diff_missing,
  cancer = cancer_missing,
  model = model_missing
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox (Intermediate concat) *",
    `cox_late_mean` = "NN Cox (Late mean) *",
    `eh_intermediate_concat` = "NN EH (Intermediate concat) *",
    `eh_late_mean` = "NN EH (Late mean) *",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "Priority elastic net"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "IBS", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[2:7]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  geom_hline(yintercept = 0, lty = 2) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  annotate("text", x = 5, y = 0.05, label = "**", size = 8) +
  labs(title = "")

diff_missing <- -(multimodal_metrics_missing %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(d_calibration)) %>% pull(mean) - multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(d_calibration)) %>% pull(mean))

cancer_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(cancer)

model_missing <- multimodal_metrics %>%
  filter(modalities == "full") %>%
  filter(model %in% c("priority_elastic_net", "cox_intermediate_concat", "cox_late_mean", "eh_intermediate_concat", "eh_late_mean")) %>%
  arrange(desc(model), desc(cancer)) %>%
  group_by(model, cancer, project) %>%
  summarise(mean = mean(antolini_concordance)) %>%
  pull(model)

c <- data.frame(
  diff = diff_missing,
  cancer = cancer_missing,
  model = model_missing
) %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC *",
    `cox_late_mean` = "NN Cox LM *",
    `eh_intermediate_concat` = "NN EH IC *",
    `eh_late_mean` = "NN EH LM *",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  )) %>%
  ggplot(aes(x = model, y = diff, fill = model)) +
  geom_violin(draw_quantiles = c(0.5)) +
  theme_big_simple() +
  labs(x = "", y = "D-CAL", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[2:7]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  geom_hline(yintercept = 0, lty = 2) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  geom_point(position = position_dodge(width = 0.75), show.legend = FALSE) +
  stat_summary(fun.y = mean, geom = "point", aes(color = "Mean"), size = 3, show.legend = FALSE) +
  scale_colour_manual(
    values = c("red"),
    name = ""
  ) +
  annotate("text", x = 5, y = 6, label = "**", size = 8) +
  labs(title = "")


figure_4_lower <- cowplot::plot_grid(a + theme(legend.position = "none"), b + theme(legend.position = "none"), c + theme(legend.position = "none"), ncol = 3, labels = c(""))

cowplot::plot_grid(figure_4_upper, figure_4_lower, legend_saved, labels = c("A", "B", ""), nrow = 3, rel_heights = c(0.5, 0.5, 0.1), label_size = 24)

ggsave("./figures_reproduced/survboard_final_fig_4.pdf", width = 18, height = 7 * 2)
ggsave("./figures_reproduced/survboard_final_fig_4.svg", width = 18, height = 7 * 2)
