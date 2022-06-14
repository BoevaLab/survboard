library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3tuning)
library(paradox)
library(mlr3proba)
library(rjson)
library(dplyr)
future::plan("multisession")
options(future.globals.onReference = "warning")
config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

remove_constants <- po("removeconstants")
set.seed(42)

learners <- list(
  remove_constants %>>% po("learner",
    id = "ranger",
    learner = lrn("surv.coxph",
      encapsulate = c(train = "evaluate", predict = "evaluate"),
      fallback = lrn("surv.kaplan")
    )
  )
)

bmr <- benchmark(benchmark_grid(tsk("rats"), learners, rsmp("cv", folds = 2)))
print(bmr)