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
  unimodal_metrics %>% filter(is.na(d_calibration)) %>% group_by(model, modalities) %>% summarise(n = n()),
  multimodal_metrics %>% filter(is.na(d_calibration)) %>% group_by(model, modalities) %>% summarise(n = n()),
  pancancer_metrics %>% filter(is.na(d_calibration)) %>% group_by(model, modalities) %>% summarise(n = n()),
  multimodal_metrics_missing %>% filter(is.na(d_calibration)) %>% group_by(model, modalities) %>% summarise(n = n()),
) %>% write_csv("./tables_reproduced/survboard_final_table_S4.csv")
