#!/usr/bin/env bash

for model in naive mean  
do
		python ../python/scripts/driver_all.py \
    		../../data \
    		../../config/config.json ../../config/params.json \
    		../../results_reproduced/pancancer/${model}/TCGA \
    		${model} TCGA pancancer multimodal_dropout
done


