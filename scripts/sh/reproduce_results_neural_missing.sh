#!/usr/bin/env bash

for model in naive mean  
do
	for project in TCGA TARGET ICGC
	do
			python ../python/scripts/driver_all.py \
    			../../data \
    			../../config/config.json ../../config/params.json \
    			../../results_reproduced/missing/${model}/${project} \
    			${model} ${project} missing multimodal_dropout
	done
done


