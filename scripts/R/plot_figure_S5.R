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


unimodal_metrics <- vroom::vroom("./unimodal_metrics_reproduced/metrics_survboard_finalized_unimodal.csv")[, -1]
misc_information <- vroom::vroom("./tables_reproduced/survboard_final_table_S2.csv")
misc_information$id <- paste0(misc_information$project, misc_information$cancer)

misc_information <- misc_information %>% select(id, n, e, p)

plt_frame <- unimodal_metrics %>% filter(model %in% c("rsf", "cox_early")) %>% select(cancer, project, model, antolini_concordance, split, modalities)
plt_frame$id <- paste0(plt_frame$project, plt_frame$cancer)
plt_frame %>% select(-project, -cancer) %>% 
  group_by(id, model) %>% summarise(antolini_concordance = mean(antolini_concordance))  %>%
  left_join(misc_information) %>% 
  pivot_wider(names_from = model, values_from = antolini_concordance) -> inter_frame
inter_frame$grp <- cut_number(inter_frame$e, 10)

a <- inter_frame %>% group_by(grp) %>% summarise(mean=mean(rsf - cox_early)) %>%
  
  ggplot(aes(x = grp, y = mean)) + geom_bar(stat="identity") + theme_simple() + labs(y = "Mean Antolini's C (RSF - NN)", x = "Number of events bin")

plt_frame <- unimodal_metrics %>% filter(model %in% c("elastic_net", "cox_early")) %>% select(cancer, project, model, antolini_concordance, split, modalities)
plt_frame$id <- paste0(plt_frame$project, plt_frame$cancer)
plt_frame %>% select(-project, -cancer) %>% 
  group_by(id, model) %>% summarise(antolini_concordance = mean(antolini_concordance))  %>%
  left_join(misc_information) %>% 
  pivot_wider(names_from = model, values_from = antolini_concordance) -> inter_frame
inter_frame$grp <- cut_number(inter_frame$e, 10)

b <- inter_frame %>% group_by(grp) %>% summarise(mean=mean(elastic_net - cox_early)) %>%
  
  ggplot(aes(x = grp, y = mean)) + geom_bar(stat="identity") + theme_simple() + labs(y = "Mean Antolini's C (EN - NN)", x = "Number of events bin")


cowplot::plot_grid(
  a, b, label_size = 24, labels = c("A", "B")
)

ggsave("./figures_reproduced/survboard_final_fig_S5.pdf", width = 18, height = 5)
