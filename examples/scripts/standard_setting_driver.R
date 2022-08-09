suppressPackageStartupMessages({
  library(here)
  library(argparse)
  source(here::here("examples", "scripts", "example_standard_setting.R"))
})


parser <- ArgumentParser()
parser$add_argument("--project", help = "Which project is being used. Must be one of TCGA, ICGC or TARGET.")
parser$add_argument("--cancer", help = "Which cancer is being used. Must be one of the valid cancer datasets for TCGA, ICGC, or TARGET. See our webservice or paper for details.")
parser$add_argument("--n_cores", help = "How many cores you would like to use.")
args <- parser$parse_args()

run_standard_setting_example_R(
  project = args$project,
  cancer = args$cancer,
  n_cores = args$n_cores
)
