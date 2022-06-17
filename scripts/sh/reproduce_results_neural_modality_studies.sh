#!/usr/bin/env bash

for model in naive mean  
do
	for project in TCGA TARGET ICGC
	do
		for modality in gex gex_clinical multi_omics_without_clinical 
		do
			python ../python/scripts/driver_${modality}.py \
    			../../data \
    			../../config/config.json ../../config/params.json \
    			../../results_reproduced \
    			${model} ${project} standard impute
	done
done


