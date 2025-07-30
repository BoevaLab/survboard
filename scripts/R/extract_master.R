# .libPaths(c("/cluster/customapps/biomed/boeva/dwissel/4.2", .libPaths()))
suppressPackageStartupMessages({
  library(vroom)
  library(rjson)
  library(dplyr)
  library(readr)
})

config <- rjson::fromJSON(
  #file = here::here("config", "config.json")
  file = "/Volumes/Backup/cr/survboard_revisions/config/config.json"
)

# Seeding for reproducibility.
set.seed(42)

for (project in c("METABRIC", "TCGA", "ICGC", "TARGET")) {
  # Iterate over all cancers in the project.
  for (cancer in config[[paste0(tolower(project), "_cancers")]]) {
    set.seed(42)
    # Read in complete modality sample dataset.
    data <- vroom::vroom(
      # here::here(
      #  "data_reproduced", project,
      #  paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      # )
      paste0("/Volumes/Backup/cr/survboard_revisions/final_revision/data_reproduced/", project, "/", paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = ""), collapse = "")
    )
    master <- data.frame(data[, c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))])
    master %>% write_csv(
      paste0("/Volumes/Backup/cr/survboard_revisions/final_revision/data_reproduced/", project, "/", paste0(cancer, "_master.csv", collapse = ""), collapse = "")
    )
  }
}

sessionInfo()
