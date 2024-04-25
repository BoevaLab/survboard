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
library(readr)

kaplan_meier_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_kaplan_meier.csv")[, -1] %>% arrange(cancer, split, project)
unimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1] %>% mutate(model = recode(model, `eh_early` = "eh_intermediate_concat", `cox_early` = "cox_intermediate_concat", `elastic_net` = "priority_elastic_net", `rsf` = "blockforest"))
multimodal_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")[, -1]
pancancer_metrics <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_pancancer.csv")[, -1]
multimodal_metrics_missing <- vroom::vroom("./metrics_reproduced/metrics_survboard_finalized_multimodal_missing.csv")[, -1]

rbind(
  unimodal_metrics %>% left_join(kaplan_meier_metrics, by = c("cancer" = "cancer", "project" = "project", "modalities" = "modalities", "split" = "split")) %>%
    mutate(is_km = integrated_brier_score.x == integrated_brier_score.y) %>%
    group_by(model.x, modalities) %>% summarise(failure_rate = mean(is_km)) %>% filter(failure_rate > 0) %>%
    rename(model = model.x),
  multimodal_metrics %>% left_join(kaplan_meier_metrics %>% filter(modalities == "clinical"), by = c("cancer" = "cancer", "project" = "project", "split" = "split")) %>%
    mutate(is_km = integrated_brier_score.x == integrated_brier_score.y) %>%
    group_by(model.x, modalities.x) %>% summarise(failure_rate = mean(is_km)) %>% filter(failure_rate > 0) %>%
    rename(model = model.x, modalities = modalities.x),
  pancancer_metrics %>% left_join(kaplan_meier_metrics %>% filter(modalities == "clinical"), by = c("cancer" = "cancer", "project" = "project", "split" = "split")) %>%
    mutate(is_km = integrated_brier_score.x == integrated_brier_score.y) %>%
    group_by(model.x, modalities.x) %>% summarise(failure_rate = mean(is_km)) %>% filter(failure_rate > 0) %>%
    rename(model = model.x, modalities = modalities.x),
  multimodal_metrics_missing %>% left_join(kaplan_meier_metrics %>% filter(modalities == "clinical"), by = c("cancer" = "cancer", "project" = "project", "split" = "split")) %>%
    mutate(is_km = integrated_brier_score.x == integrated_brier_score.y) %>%
    group_by(model.x, modalities.x) %>% summarise(failure_rate = mean(is_km)) %>% filter(failure_rate > 0) %>%
    rename(model = model.x, modalities = modalities.x)
) %>% write_csv("./tables_reproduced/survboard_final_table_S1.csv")
