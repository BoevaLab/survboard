#!/usr/bin/env bash

for model in naive mean  
do
	for project in TCGA TARGET ICGC
	do
		python ../python/scripts/driver_all_alpha_ablation.py \
    		../../data \
    		../../config/config.json ../../config/params.json \
    		../../results_reproduced/standard/${model}/${project}/alpha_ablation \
    		${model} ${project} standard impute
	done
done


