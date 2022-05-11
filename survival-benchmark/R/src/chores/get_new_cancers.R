library(TCGAbiolinks)
library(here)
library(rjson)
library(tibble)
library(maftools)
library(vroom)

source(here::here("survival-benchmark", "R", "src", "prep", "prepare_tcga_data.R"))
source(here::here("survival-benchmark", "R", "src", "query", "read_tcga_data.R"))
source(here::here("survival-benchmark", "R", "src", "prep", "prepare_icgc_data.R"))

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 2)
Sys.setenv(PATH = paste(c(paste0("/Users/", Sys.info()[["user"]], "/miniforge3/envs/thesis/bin"), Sys.getenv("PATH"),
  collapse = .Platform$path.sep
), collapse = ":"))

# tumors <- TCGAbiolinks::getGDCprojects() %>% pull(tumor)
# tumors <- tumors[-unname(sapply(config$cancers, function(x) grep(x, tumors)))]
#

#gex_master <- vroom(
#   here::here(
#    "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA", "EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv"
#   )
# ) %>% data.frame(check.names = FALSE)

#cnv_master <- vroom(here::here(
#  "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA", "TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
# )) %>% data.frame(check.names = FALSE)

#meth_master <- vroom(here::here(
#  "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA", "jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450#.betaValue_whitelisted.tsv"
#)) %>% data.frame(check.names = FALSE)

#mirna_master <- vroom(here::here(
#  "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA", "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.csv")) %>%
#   data.frame(check.names = FALSE)

#mutation <- maftools::read.maf(here::here(
#  "~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA",
#  "mc3.v0.2.8.PUBLIC.maf.gz"))
#mut_master <- mutCountMatrix(mutation,
#   removeNonMutated = FALSE
# )

clinical <- read_raw_clinical_data()
tcga_cdr <- clinical$clinical_data_resource_outcome
tcga_w_followup <- clinical$clinical_with_followup
#rppa_master <- vroom::vroom(here::here(
#"~", "boeva_lab_scratch", "data", "projects", "David", "Nikita_David_survival_benchmark", "survival_benchmark", "data", "raw", "TCGA", "TCGA-RPPA-pancan-clean.txt")) %>% data.frame(check.names =FALSE)

new_cancers <- c(
  "LAML",
  "CESC",
  "GBM",
  "READ"
)

icgc_cancers <- c(
  "PACA-AU",
  "CLLE-ES",
  "PACA-CA",
  "LICA-FR",
  "LIRI-JP"
)

#mut_master_icgc <- vroom::vroom(
#  here::here(
#    "~", "Downloads", "donor_SNV.donor.codingMutation-allProjects.gz"
#  )
#)
#colnames(mut_master_icgc)[1] <- "icgc_donor_id"

for (cancer in icgc_cancers) {
  prepare_icgc(cancer = cancer)
}

#for (cancer in config$cancers[8:14]) {
#prepare_new_cancer_dataset(
#    cancer=cancer, include_rppa = TRUE, standard_clinical = TRUE
#  )
#}

#prepare_new_cancer_dataset(
#  cancer = "LAML",
#  include_mutation = FALSE,
#  include_rppa = FALSE,
#  standard_clinical = TRUE
#)

#prepare_new_cancer_dataset(
#  cancer = "CESC",
#  include_mutation = TRUE,
#  include_rppa = TRUE,
#  standard_clinical = TRUE
#)

#prepare_new_cancer_dataset(
#  cancer = "GBM",
#  include_methylation = FALSE,
#  include_rppa = FALSE,
#  include_mirna = FALSE,
#  standard_clinical = TRUE
#)

#prepare_new_cancer_dataset(
#  cancer = "READ",
#  include_rppa = FALSE,
#  standard_clinical = TRUE
#)

#for (cancer in icgc_cancers) {
#  prepare_icgc(cancer)
#}
