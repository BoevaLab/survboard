plt_frame <- data.frame()
for (i in config$target_cancers) {
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "BlockForest", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "BlockForest_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "CoxBoost_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "CoxBoost", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "ranger", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "ranger_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "poe", "standard_metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TARGET", i, "mean", "standard_metrics.csv"
      )
    ), cancer = i)
  )
}
library(ggplot2)
plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = concordance, fill = model)
)  + geom_boxplot() + facet_wrap(~cancer)

plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = ibs, fill = model)
)  + geom_boxplot() + facet_wrap(~cancer)

plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = d_calib, fill = model)
)  + geom_boxplot() + facet_wrap(~cancer)

library(ggplot2)
library(dplyr)

plt_frame <- data.frame()
for (i in config$icgc_cancers) {
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "BlockForest", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "BlockForest_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "CoxBoost_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "CoxBoost", "metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "ranger", "metrics.csv"
      )
    ), cancer = i)
  )

  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "poe", "standard_metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "mean", "standard_metrics.csv"
      )
    ), cancer = i)
  )
  
  plt_frame <- rbind(
    plt_frame,
    cbind(readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "ICGC", i, "ranger_favoring", "metrics.csv"
      )
    ), cancer = i)
  )
}
library(ggplot2)
plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = concordance, fill = model)
)  + geom_boxplot() + facet_wrap(~cancer, scales = "free")

Pplt_frame %>% ggplot2::ggplot(
  aes(x = model, y = ibs, fill = model)
)  + geom_boxplot() +f

plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = d_calib, fill = model)
)  + geom_boxplot()


plt_frame <- data.frame()
for (i in c("BLCA")) {
  plt_frame <- rbind(
    plt_frame,
    readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TCGA", i, "dae", "standard_metrics.csv"
      )
    )
  )
  
  plt_frame <- rbind(
    plt_frame,
    readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TCGA", i, "poe", "standard_metrics.csv"
      )
    )
  )
  
  plt_frame <- rbind(
    plt_frame,
    readr::read_csv(
      here::here(
        "data", "results", "survival_functions", "TCGA", i, "mean", "standard_metrics.csv"
      )
    )
  )
}
library(ggplot2)
plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = concordance, fill = model)
)  + geom_boxplot()

plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = ibs, fill = model)
)  + geom_boxplot()

plt_frame %>% ggplot2::ggplot(
  aes(x = model, y = d_calib, fill = model)
)  + geom_boxplot()
