#!/bin/sh

python ../python/examples/scripts/standard_setting.py \
    ../../data \ # Root directory of data. Download the data and pass the path to that folder.
    ../../config/config.json \ # Path to the config file which contains list of cancers for each project. To test for a subset of cancers simply alter the list accordingly.
    ../../config/params.json \ # Path to the parameter file for the model to be tested. 
    ../../results_reproduced/standard/ \ # Path where the results should be saved
    TCGA standard impute # Last argument not applicable for standard setting since there is no missing, hence passing default 'impute' as a value.

# For other project, replace TCGA by the respective project name.
# Similarly, for specific cancers, open the provided config.json file in the config folder and specify the cancers to test
# To test other setings, replace 'standard' by 'missing' or 'pancancer'.
# For missing modality handling, can use either 'impute' or 'multimodal_dropout' if using one of our models,
# or can specify a custom way of handling for your model and adapt the script accordingly.
