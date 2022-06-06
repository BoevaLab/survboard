library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
#library(mlr3tuning)
#library(paradox)
library(mlr3proba)
#library(rjson)
#library(dplyr)
future::plan("multisession")
options(future.globals.onReference = "error")
#config <- rjson::fromJSON(
#  file = here::here("config", "config.json")
#)

# select the multisession backend
future::plan("multisession")

task = tsk("spam")
learner = lrn("classif.rpart")
resampling = rsmp("subsampling")

time = Sys.time()
resample(task, learner, resampling)
Sys.time() - time