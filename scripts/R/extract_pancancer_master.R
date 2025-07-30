suppressPackageStartupMessages({
  library(readr)
  library(vroom)
  library(dplyr)
})


set.seed(42)

data <- vroom::vroom(
  here::here(
    "data_reproduced", "TCGA", "pancancer_complete.csv"
  )
)

data %>%
  dplyr::select(
    OS,
    OS_days,
    clinical_cancer_type
  ) %>%
  write_csv(here::here(
    "data_reproduced", "TCGA", "pancancer_complete_master.csv"
  ))

