library(here)
library(rjson)
library(tibble)
library(maftools)
library(vroom)

source(here::here("survboard", "R", "prep", "prepare_tcga_data.R"))
source(here::here("survboard", "R", "prep", "prepare_icgc_data.R"))
source(here::here("survboard", "R", "prep", "read_tcga_data.R"))

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 2)

gex_master <- vroom(
  here::here(
    "data_template", "TCGA", "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
  )
) %>% data.frame(check.names = FALSE)

cnv_master <- vroom(here::here(
  "data_template", "TCGA", "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
)) %>% data.frame(check.names = FALSE)

meth_master <- vroom(here::here(
  "data_template", "TCGA", "jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv"
)) %>% data.frame(check.names = FALSE)

mirna_master <- vroom(here::here(
  "data_template", "TCGA", "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv"
)) %>%
  data.frame(check.names = FALSE)

mutation <- maftools::read.maf(here::here(
  "data_template", "TCGA",
  "mc3.v0.2.8.PUBLIC.maf.gz"
))
mut_master <- mutCountMatrix(mutation,
  removeNonMutated = FALSE
)

clinical <- read_raw_clinical_data()
tcga_cdr <- clinical$clinical_data_resource_outcome
tcga_w_followup <- clinical$clinical_with_followup
rppa_master <- vroom::vroom(here::here(
  "data_template", "TCGA", "TCGA-RPPA-pancan-clean.txt"
)) %>% data.frame(check.names = FALSE)

non_full_cancers <- c(
  "LAML",
  "CESC",
  "GBM",
  "READ",
  "SKCM"
)

for (cancer in config$tcga_cancers[!(config$tcga_cancers %in% non_full_cancers)]) {
  prepare_new_cancer_dataset(
    cancer = cancer, include_rppa = TRUE, standard_clinical = TRUE
  )
}

prepare_new_cancer_dataset(
  cancer = "LAML",
  include_mutation = FALSE,
  include_rppa = FALSE
)

prepare_new_cancer_dataset(
  cancer = "CESC",
  include_mutation = TRUE,
  include_rppa = TRUE,
  standard_clinical = TRUE
)

prepare_new_cancer_dataset(
  cancer = "GBM",
  include_methylation = FALSE,
  include_rppa = FALSE,
  include_mirna = FALSE,
  standard_clinical = TRUE
)

prepare_new_cancer_dataset(
  cancer = "READ",
  include_rppa = FALSE,
  standard_clinical = TRUE
)

prepare_new_cancer_dataset(
  cancer = "SKCM", include_rppa = FALSE, standard_clinical = TRUE,
  include_gex = TRUE, include_mirna = FALSE, include_mutation = TRUE,
  include_methylation = TRUE, include_cnv = FALSE
)

for (cancer in config$icgc_cancers) {
  prepare_icgc(cancer)
}
