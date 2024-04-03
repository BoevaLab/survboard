suppressPackageStartupMessages({
  library(here)
  library(readr)
  library(vroom)
  library(readxl)
})


#' Helper function to read raw clinical files (since some information)
#' is only available in CDR and some only in the TSV.

#' @returns list. List of length 2, where the first element contains
#' a data.frame containing the CDR information and the second element
#' contains a data.frame containing the TSV information.
read_raw_clinical_data <- function() {
  list(
    clinical_data_resource_outcome = readxl::read_xlsx(
      here::here("data_template", "TCGA", "TCGA-CDR-SupplementalTableS1.xlsx"),
      guess_max = 2500,
      range = cell_cols("B:AH")
    ),
    clinical_with_followup = read_tsv(
      here::here(
        "data_template", "TCGA", "clinical_PANCAN_patient_with_followup.tsv"
      ),
      guess_max = 1e5
    )
  )
}
