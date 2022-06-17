# SurvBoard: Standardised Benchmarking for Multi-omics Cancer Survival Models
This Github repository contains documentation, resources, links and code for our manuscript "SurvBoard: Standardised Benchmarking for Multi-omics Cancer Survival Models".

## Summary
SurvBoard is a benchmark focused on evaluating multi-omics survival methods. Although in principle other methods (e.g., models predicting survival only from gene expression data) are applicable to SurvBoard, our primary focus were models that incorporate multi-omics data in conjuction with clinical variables. Thus, our web service currently only enables submission for multi-omics methods, but we are working on extending SurvBoard also to single-omics and other methods using arbitary omics information. 

For more information, please find links and a reproduction guide below. You may also refer to our paper for further details on all aspects of our work.

## Resources
### Links
Paper - Our manuscript is still under review.

[Web service](https://survboard.vercel.app/)

[Data, splits and benchmark results (both metrics and full survival functions for all splits)](https://ibm.ent.box.com/v/survboard-meta)

### Reproduction guide
To reproduce our results, first install the needed `R` and `python` packages:

```r
install.packages(c("renv", "here"))
library(renv)
library(here)
renv::restore(here::here())
```

and 

```
python -m venv venv
pip install -e .
```



Afterward, run the following bash script to reproduce all of our results:

```
bash reproduce.sh
```

Also, you may repeat our preprocessing by running:

```
bash preprocess.sh
```

Please note that 

All reproduced results will be written to a newly created `results_reproduced` which will have the following structure (similar to our own results, obtainable from the links above):

```
survival_benchmark
└───results_reproduced
       survival_functions
        └────TCGA
            └───BLCA
                └───BlockForest
                    └───split_1.csv
                ...
            ...
        ...
```

Please refer to our full results to get a better idea of what the resulting data will look like. Please also note that all "raw" output files generated during the benchmark (e.g., `.rds` from `R` used for storing intermediate results) will be cleaned up by this command and you will "only" be left with the folder detailed above.

## License
Please refer to the respective models and datasets mentioned above for their licenses. All of our own work (including this repository) is under a MIT license.

## Questions & feedback
In case you have any feedback, questions or other issues related to either our manuscript or the code found within this repository, please contact us at NJA@zurich.ibm.com, aayush.grover@ethz.ch and david.wissel@ethz.ch.
