library(ggpubfigs)
library(dplyr)
library(caret)
library(survival)
library(survminer)
library(survcomp)
library(tidyr)
library(pheatmap)
library(RColorBrewer)

clin <- readxl::read_xlsx("./data_template/TCGA/tcga_clinical.xlsx")[, -1]
followup <- vroom::vroom("./data_template/TCGA/clinical_PANCAN_patient_with_followup.tsv")

clin %>%
  filter(type == "BLCA") %>%
  filter(!is.na(OS)) %>%
  filter(!is.na(OS.time)) %>%
  left_join(followup %>% dplyr::select(bcr_patient_barcode, radiation_therapy)) -> clinical_blca


multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]

multimodal_metrics <- multimodal_metrics %>%
  group_by(project, cancer, model, modalities) %>%
  mutate(d_calibration = replace_na(d_calibration, mean(d_calibration, na.rm = TRUE))) %>%
  ungroup() %>%
  mutate(model = recode(
    model,
    `cox_intermediate_concat` = "NN Cox IC",
    `cox_late_mean` = "NN Cox LM",
    `eh_intermediate_concat` = "NN EH IC",
    `eh_late_mean` = "NN EH LM",
    `blockforest` = "BlockForest",
    `priority_elastic_net` = "PriorityLasso L1+L2"
  ))


tmp <- multimodal_metrics %>% filter(modalities == "clinical_gex" & cancer == "BRCA" & project == "METABRIC")
tmp <- cor(tmp[, 6:8])
colnames(tmp) <- c("Antolini's C", "IBS", "D-CAL")
rownames(tmp) <- colnames(tmp)
a_a_heatmap <- pheatmap(tmp,
  cluster_cols = FALSE,
  cluster_rows = FALSE,
  show_colnames = TRUE,
  fontsize_row = 24,
  fontsize_col = 24,
  fontsize = 20,
  legend = FALSE,
  show_rownames = TRUE,
  color = colorRampPalette(brewer.pal(n = 7, name = "PuBuGn"))(10),
  scale = "none",
  display_numbers = TRUE
)

a_a <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & cancer == "BRCA" & project == "METABRIC") %>%
  ggplot(aes(x = model, y = 1 - antolini_concordance, fill = model)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "1 - Antolini's C", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))



a_b <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & cancer == "BRCA" & project == "METABRIC") %>%
  ggplot(aes(x = model, y = integrated_brier_score, fill = model)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "IBS", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))


a_c <- multimodal_metrics %>%
  filter(modalities == "clinical_gex" & cancer == "BRCA" & project == "METABRIC") %>%
  ggplot(aes(x = model, y = d_calibration, fill = model)) +
  geom_boxplot() +
  theme_big_simple() +
  labs(x = "", y = "D-CAL", fill = "") +
  scale_fill_manual(values = ggpubfigs::friendly_pals$muted_nine) +
  scale_fill_manual(values = ggpubfigs::friendly_pals$ito_seven[1:6]) +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10)))


fig_5_a <- cowplot::plot_grid(a_a + theme(legend.position = "none"),
  a_b + theme(legend.position = "none"),
  a_c + theme(legend.position = "none"),
  ncol = 3,
  rel_widths = c(1, 1, 1)
)
fig_5_a <- cowplot::plot_grid(fig_5_a, cowplot::get_legend(a_a), nrow = 2, rel_heights = c(0.9, 0.2))

clin %>%
  filter(type == "GBM") %>%
  filter(!is.na(OS)) %>%
  filter(!is.na(OS.time)) %>%
  left_join(followup %>% dplyr::select(bcr_patient_barcode, radiation_therapy)) %>%
  filter(radiation_therapy %in% c("NO", "YES")) -> clinical_blca


fit <- survfit(Surv(OS.time, OS) ~ radiation_therapy, data = clinical_blca)

fig_5_d_two <- ggsurvplot(fit,
  data = clinical_blca, pval = FALSE, legend.labs =
    c("Untreated", "Treated"),
  palette =
    ggpubfigs::friendly_pals$ito_seven[1:2],
  size = 1,
  pval.coord = c(1000, 1.0),
  font.x = c(24),
  font.y = c(24),
  legend.title = ""
)

fig_5_d_two <- ggpar(fig_5_d_two,
  font.main = c(24),
  font.x = c(20),
  font.y = c(20),
  font.caption = c(14),
  font.legend = c(18),
  font.tickslab = c(14)
)


fig_5_d_two$plot +
  annotate("text", x = 1500, y = 1.0, label = "p < 0.0001", size = 7)


km <- fig_5_d_two$plot +
  annotate("text", x = 2000, y = 1.0, label = "p < 0.0001", size = 7) + theme(legend.position = "bottom") + theme_big_simple() +
  theme(
    axis.ticks.x = element_line(NULL)
  ) +
  theme(legend.text = element_text(size = 24)) +
  guides(fill = guide_legend(override.aes = list(size = 10))) +
  labs(y = expression(P(t * " | " * X)), x = "t (days)") +
  scale_x_continuous(breaks = c(0, 1500, 3000))


first_row_fig_5 <- cowplot::plot_grid(NULL, a_a_heatmap$gtable, fig_5_a, cowplot::plot_grid(km + theme(legend.position = "none"), get_legend(km), nrow = 2, rel_heights = c(0.9, 0.2)), ncol = 4, rel_widths = c(0.05, 0.4, 1, 0.4), labels = c("A", "", "B", "C"), label_size = 24)



multimodal_metrics %>% filter(cancer %in% c("BRCA", "PAAD", "ALL", "PACA-AU", "CLLE-ES") & project %in% c("TCGA", "TARGET", "ICGC") & modalities == "clinical_gex") -> fig_5_b_plt_frame

fig_5_b_plt_frame$cv_ix <- rep(rep(1:5, each = 5), 30)

fig_5_b_plt_frame_other <- fig_5_b_plt_frame
fig_5_b_plt_frame_other$cv_ix <- "All"


colorRamp(c(
  "blue4",
  "khaki1"
))(c((1:6) / 6))


myColors <- c(
  "blue4",
  "khaki1"
)
myRangeFunction <- colorRampPalette(myColors)
myColorRange <- myRangeFunction(6)
myColors <- myColorRange[1:6]

fig_5_b <- rbind(
  fig_5_b_plt_frame %>% group_by(cv_ix, modalities, model, cancer, project) %>% summarise(mean = mean(antolini_concordance)) %>% group_by(cv_ix, modalities, cancer, project) %>% mutate(rank = rank(-mean)) %>%
    ungroup(),
  fig_5_b_plt_frame_other %>% group_by(cv_ix, modalities, model, cancer, project) %>% summarise(mean = mean(antolini_concordance)) %>% group_by(cv_ix, modalities, cancer, project) %>% mutate(rank = rank(-mean)) %>%
    ungroup()
) %>%
  mutate(cancer = recode(
    cancer,
    `ALL` = "TARGET-ALL",
    `BRCA` = "TCGA-BRCA",
    `CLLE-ES` = "ICGC-CLLE-ES",
    `LUSC` = "TCGA-LUSC",
    `PAAD` = "TCGA-PAAD",
    `PACA-AU` = "ICGC-PACA-AU"
  )) %>%
  ggplot(aes(x = cv_ix, y = model, fill = factor(rank))) +
  geom_tile() +
  facet_wrap(~cancer, nrow = 1) +
  theme_bw(base_size = 24) +
  scale_fill_manual(values = rev(myColors)) +
  labs(x = "CV index", y = "", fill = "Antolini's C Rank")

cowplot::plot_grid(first_row_fig_5, fig_5_b, nrow = 2, labels = c("", "D"), label_size = 24)


first_row_fig_5 <- cowplot::plot_grid(NULL, a_a_heatmap$gtable, fig_5_a, ncol = 3, rel_widths = c(0.05, 0.4, 1), label_size = 24)

ggsave("~/Downloads/survboard_seminar_002.pdf", plot=first_row_fig_5, width = 18, height = 13 / 1.5 / 2)

ggsave("./figures_reproduced/survboard_final_fig_5.pdf", width = 18, height = 13 / 1.5)
ggsave("./figures_reproduced/survboard_final_fig_5.svg", width = 18, height = 13 / 1.5)
