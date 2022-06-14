library(here)
library(readr)
library(vroom)
library(readxl)

read_raw_clinical_data <- function() {
  list(
    clinical_data_resource_outcome = readxl::read_xlsx(
      here::here("data", "raw", "TCGA", "TCGA-CDR-SupplementalTableS1.xlsx"),
      guess_max = 2500,
      range = cell_cols("B:AH")
    ),
    clinical_with_followup = read_tsv(
      here::here(
        "data", "raw", "TCGA", "clinical_with_followup.tsv"
      ),
      guess_max = 1e5
    )
  )
}
