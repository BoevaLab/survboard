#!/usr/bin/env bash

for model in naive mean  
do
	for project in TCGA TARGET ICGC
	do
		for handle_missing in multimodal_dropout
		do
			python ../python/scripts/driver_all.py \
    			../../data \
    			../../config/config.json ../../config/params.json \
    			../../results_reproduced \
    			${model} ${project} pancancer ${handle_missing}
	done
done


