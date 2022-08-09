## Overview
This folder contains examples for each setting for both R and Python. The goal of these examples is two-fold:

- Make submission to our webservice easier by providing code such that researchers only have to implement their models
- Allow for reproduction of model submissions by us

### Accessibility
#### R
Please note that `example_missing_setting.R` and `example_pancancer_setting.R` serve only as examples and are **not** functional, since we are not aware of existing methods in R which handle missing modality samples. That said, if you implement 

We assume that you will implement your model in [mlr3](https://github.com/mlr-org/mlr3) - once your model itself is written, this only requires a few additional lines and makes standardized testing for us much easier. Please reach out or open an issue on Github in case you have any questions.

The `train_all_R.sh` bash script will call the respective benchmarking function - you may simply change the respective call in `train_all_R.sh` to reflect the setting in which you are interested.

#### Python
TODO

### Reproduction
For reproduction, please submit the following (provided through a Github or other online repository link):

- The `examples` folder - please rename this to `submission` 
- Please remove any files not needed for your submission
- Only keep: (i) The respective `train_all.sh` bash script that can be used to reproduce your predictions and (ii) any code needed by your models (please use the code structure of our example files, although you may of course rename them) (iii) You may assume that any code in our original Github repo will be available within the same location when we reproduce your submission.