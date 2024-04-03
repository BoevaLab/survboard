library(dplyr)
library(rjson)

config <- rjson::fromJSON(
  file = here::here(
    #"//Volumes",
    #"Backup",
    #"transfer",
    #"20231123",
    #"survboard",
    "config", "config.json")
)


complete_list <- list()
incomplete_list <- list()


for (cancer in config$tcga_cancers) {
  data <- vroom::vroom(
    here::here(
      "data_reproduced", "TCGA",

      paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
    )
  )
  #data <- data.frame(data[, -which("clinical_patient_id" == colnames(data))], check.names = FALSE)
  complete_list[[cancer]] <- data
  
  data <- vroom::vroom(
    here::here(
      "data_reproduced", "TCGA",
      paste0(cancer, "_data_incomplete_modalities_preprocessed.csv", collapse = "")
    )
  )
  
  #data <- data.frame(data[, -which("patient_id" == colnames(data))], check.names = FALSE)
  incomplete_list[[cancer]] <- data
}
complete_cancers <- config$tcga_cancers[-grep("(SKCM|LAML|GBM|READ)", config$tcga_cancers)]
joint_features <- Reduce(intersect, lapply(complete_list[complete_cancers], function(x) colnames(x)))

complete_list[["SKCM"]][, grep("(cnv|rppa|mirna)", joint_features, value = TRUE)] <- NA
incomplete_list[["SKCM"]][, grep("(cnv|rppa|mirna)", joint_features, value = TRUE)] <- NA
complete_list[["LAML"]][, grep("(mutation|rppa)", joint_features, value = TRUE)] <- NA
incomplete_list[["LAML"]][, grep("(mutation|rppa)", joint_features, value = TRUE)] <- NA
complete_list[["GBM"]][, grep("(meth|rppa|mirna)", joint_features, value = TRUE)] <- NA
incomplete_list[["GBM"]][, grep("(meth|rppa|mirna)", joint_features, value = TRUE)] <- NA
complete_list[["READ"]][, grep("rppa", joint_features, value = TRUE)] <- NA
incomplete_list[["READ"]][, grep("rppa", joint_features, value = TRUE)] <- NA




for (cancer in config$tcga_cancers) {
  complete_list[[cancer]][, "clinical_cancer_type"] <- cancer
  incomplete_list[[cancer]][, "clinical_cancer_type"] <- cancer
}

joint_features_all <- Reduce(intersect, lapply(complete_list, function(x) colnames(x)))
complete_pancancer <- data.frame(matrix(data = NA, nrow = sum(sapply(complete_list, nrow)), ncol = length(joint_features_all)))
colnames(complete_pancancer) <- joint_features_all
incomplete_pancancer <- data.frame(matrix(data = NA, nrow = sum(sapply(incomplete_list, nrow)), ncol = length(joint_features_all)))
colnames(incomplete_pancancer) <- joint_features_all
library(data.table)



complete_pancancer <- data.table::rbindlist(lapply(complete_list, function(x) x %>% select(all_of(joint_features_all))))
incomplete_pancancer <- data.table::rbindlist(lapply(incomplete_list, function(x) x %>% select(all_of(joint_features_all))))

complete_pancancer_backup <- complete_pancancer
incomplete_pancancer_backup <- incomplete_pancancer



library(readr)
complete_pancancer <- data.frame(complete_pancancer, check.names = FALSE)
incomplete_pancancer <- data.frame(incomplete_pancancer, check.names = FALSE)

complete_pancancer %>% write_csv(
  here::here(
    "data_reproduced" , "TCGA", "pancancer_complete.csv"
  )
)

incomplete_pancancer %>% write_csv(
  here::here(
    "data_reproduced", "TCGA", "pancancer_incomplete.csv"
  )
)
