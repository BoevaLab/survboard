# SurvBoard: Standardised Benchmarking for Multi-omics Cancer Survival Models
This Github repository contains documentation, resources, links, and code for our manuscript "SurvBoard: Standardised Benchmarking for Multi-omics Cancer Survival Models".

## Summary
SurvBoard is a benchmark focused on evaluating multi-omics survival methods. Although in principle other methods (e.g., models predicting survival only from gene expression data) can be applied to SurvBoard, our primary focus was on models that incorporate multi-omics data in conjunction with clinical variables. Thus, our web service currently only enables submission for multi-omics methods, but we are working on extending SurvBoard also to single-omics and other methods using arbitrary omics information. 

For more information, please find links and a reproduction guide below. You may also refer to our paper for further details on all aspects of our work.

## Resources
### Links
Paper - Our manuscript is still under review.

[Web service](https://survboard.science/)

[Raw and preprocessed data, splits, and benchmark results](https://zenodo.org/records/11066227)

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
pip install -e . # -e signifies installation in "editable" mode. Remove for normal installation.
```

Afterward, you may repeat our preprocessing by running:


```
bash scripts/sh/reproduce_preprocessing.sh
```

When re-running preprocessing, files will be written to the `data_reproduced` folder. We require certain raw data files which can be obtained from the official TCGA/ICGC/TARGET websites or from [cBioPortal](https://www.cbioportal.org/). We expect raw data files in the `data_template` folder, within their requisite project subfolder. Please see the respective preprocessing scripts for more details on file names, or contact us if you run into any issues.

Our results may be reproduced using:

```
bash scripts/sh/reproduce_experiments.sh
```

And our figures and tables using:

```
bash scripts/sh/reproduce_figures_and_tables.sh
```

All reproduced results will be written to a newly created `results_reproduced` which will have the following structure (similar to our results, obtainable from the links above).

```
survboard
└───results_reproduced
       survival_functions
       └────experiment_type
            └────TCGA
                └───BLCA
                    └───BlockForest
                        └───split_1.csv
                ...
            ...
        ...
```

Please refer to our full results to get a better idea of what the resulting data will look like.


## License
Please refer to the respective models and datasets mentioned above for their licenses. All of our work (including this repository) is under an MIT license.

## Questions & feedback
In case you have any feedback, questions, or other issues related to either our manuscript or the code found within this repository, please contact us at NJA@zurich.ibm.com, aayush.grover@ethz.ch, and dwissel@ethz.ch.
