rule all:
    input:
        expand(
            "results_reproduced/timings/prioritylasso_{cancer}",
            cancer=[
                "BLCA",
                "BRCA",
                "COAD",
                "ESCA",
                "HNSC",
                "KIRC",
                "KIRP",
                "LGG",
                "LUAD",
                "PAAD",
                "SARC",
                "SKCM",
                "STAD",
                "UCEC",
                "OV",
                "LIHC",
                "LUSC",
                "LAML",
                "CESC",
                "GBM",
                "READ",
            ],
        ),
        expand(
            "results_reproduced/timings/blockforest_{cancer}",
            cancer=[
                "BLCA",
                "BRCA",
                "COAD",
                "ESCA",
                "HNSC",
                "KIRC",
                "KIRP",
                "LGG",
                "LUAD",
                "PAAD",
                "SARC",
                "SKCM",
                "STAD",
                "UCEC",
                "OV",
                "LIHC",
                "LUSC",
                "LAML",
                "CESC",
                "GBM",
                "READ",
            ],
        ),
        expand(
            "results_reproduced/timings/{fusion}_eh_{cancer}",
            cancer=[
                "BLCA",
                "BRCA",
                "COAD",
                "ESCA",
                "HNSC",
                "KIRC",
                "KIRP",
                "LGG",
                "LUAD",
                "PAAD",
                "SARC",
                "SKCM",
                "STAD",
                "UCEC",
                "OV",
                "LIHC",
                "LUSC",
                "LAML",
                "CESC",
                "GBM",
                "READ",
            ],
            fusion=[
                "late_mean",
                "intermediate_concat",
            ],
        ),
        expand(
            "results_reproduced/timings/{fusion}_cox_{cancer}",
            cancer=[
                "BLCA",
                "BRCA",
                "COAD",
                "ESCA",
                "HNSC",
                "KIRC",
                "KIRP",
                "LGG",
                "LUAD",
                "PAAD",
                "SARC",
                "SKCM",
                "STAD",
                "UCEC",
                "OV",
                "LIHC",
                "LUSC",
                "LAML",
                "CESC",
                "GBM",
                "READ",
            ],
            fusion=[
                "late_mean",
                "intermediate_concat",
            ],
        ),
        expand(
            "results_reproduced/timings/{fusion}_{cancer}",
            cancer=[
                "BLCA",
                "BRCA",
                "COAD",
                "ESCA",
                "HNSC",
                "KIRC",
                "KIRP",
                "LGG",
                "LUAD",
                "PAAD",
                "SARC",
                "SKCM",
                "STAD",
                "UCEC",
                "OV",
                "LIHC",
                "LUSC",
                "LAML",
                "CESC",
                "GBM",
                "READ",
            ],
            fusion=[
                "gdp",
                "salmon",
                "make_salmon_data",
                "denoising_ae",
                "survival_net",
                "multimodal_survival_pred",
                "customics",
            ],
        ),


rule time_time_prioritylasso:
    output:
        "results_reproduced/timings/prioritylasso_{cancer}",
    log:
        "logs/timings/prioritylasso_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/prioritylasso_{cancer}.tsv", 5)
    script:
        "scripts/R/time_prioritylasso.R"


rule time_time_blockforest:
    output:
        "results_reproduced/timings/blockforest_{cancer}",
    log:
        "logs/timings/blockforest_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/blockforest_{cancer}.tsv", 5)
    script:
        "scripts/R/time_blockforest.R"


rule time_time_early_discrete_time:
    output:
        "results_reproduced/timings/early_discrete_time_{cancer}",
    log:
        "logs/timings/early_discrete_time_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/early_discrete_time_{cancer}.tsv", 5)
    script:
        "scripts/python/time_early_discrete_time.py"


rule time_time_late_mean_discrete_time:
    output:
        "results_reproduced/timings/late_mean_discrete_time_{cancer}",
    log:
        "logs/timings/late_mean_discrete_time_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/late_mean_discrete_time_{cancer}.tsv", 5)
    script:
        "scripts/python/time_late_mean_discrete_time.py"


rule time_time_intermediate_concat_discrete_time:
    output:
        "results_reproduced/timings/intermediate_concat_discrete_time_{cancer}",
    log:
        "logs/timings/intermediate_concat_discrete_time_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/intermediate_concat_discrete_time_{cancer}.tsv", 5)
    script:
        "scripts/python/time_intermediate_concat_discrete_time.py"


rule time_time_intermediate_attention_discrete_time:
    output:
        "results_reproduced/timings/intermediate_attention_discrete_time_{cancer}",
    log:
        "logs/timings/intermediate_attention_discrete_time_{cancer}.log",
    threads: 1
    benchmark:
        repeat(
            "benchmarks/timings/intermediate_attention_discrete_time_{cancer}.tsv", 5
        )
    script:
        "scripts/python/time_intermediate_attention_discrete_time.py"


rule time_time_early_cox:
    output:
        "results_reproduced/timings/early_cox_{cancer}",
    log:
        "logs/timings/early_cox_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/early_cox_{cancer}.tsv", 5)
    script:
        "scripts/python/time_early_cox.py"


rule time_time_late_mean_cox:
    output:
        "results_reproduced/timings/late_mean_cox_{cancer}",
    log:
        "logs/timings/late_mean_cox_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/late_mean_cox_{cancer}.tsv", 5)
    script:
        "scripts/python/time_late_mean_cox.py"


rule time_time_intermediate_concat_cox:
    output:
        "results_reproduced/timings/intermediate_concat_cox_{cancer}",
    log:
        "logs/timings/intermediate_concat_cox_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/intermediate_concat_cox_{cancer}.tsv", 5)
    script:
        "scripts/python/time_intermediate_concat_cox.py"


rule time_time_intermediate_attention_cox:
    output:
        "results_reproduced/timings/intermediate_attention_cox_{cancer}",
    log:
        "logs/timings/intermediate_attention_cox_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/intermediate_attention_cox_{cancer}.tsv", 5)
    script:
        "scripts/python/time_intermediate_attention_cox.py"


rule time_time_early_eh:
    output:
        "results_reproduced/timings/early_eh_{cancer}",
    log:
        "logs/timings/early_eh_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/early_eh_{cancer}.tsv", 5)
    script:
        "scripts/python/time_early_eh.py"


rule time_time_late_mean_eh:
    output:
        "results_reproduced/timings/late_mean_eh_{cancer}",
    log:
        "logs/timings/late_mean_eh_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/late_mean_eh_{cancer}.tsv", 5)
    script:
        "scripts/python/time_late_mean_eh.py"


rule time_time_intermediate_concat_eh:
    output:
        "results_reproduced/timings/intermediate_concat_eh_{cancer}",
    log:
        "logs/timings/intermediate_concat_eh_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/intermediate_concat_eh_{cancer}.tsv", 5)
    script:
        "scripts/python/time_intermediate_concat_eh.py"


rule time_time_intermediate_attention_eh:
    output:
        "results_reproduced/timings/intermediate_attention_eh_{cancer}",
    log:
        "logs/timings/intermediate_attention_eh_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/intermediate_attention_eh_{cancer}.tsv", 5)
    script:
        "scripts/python/time_intermediate_attention_eh.py"


### NEW STUFF


rule time_time_salmon:
    output:
        "results_reproduced/timings/salmon_{cancer}",
    log:
        "logs/timings/salmon_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/salmon_{cancer}.tsv", 5)
    script:
        "scripts/python/time_salmon.py"


rule time_time_multimodal_survival_pred:
    output:
        "results_reproduced/timings/multimodal_survival_pred_{cancer}",
    log:
        "logs/timings/multimodal_survival_pred_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/multimodal_survival_pred_{cancer}.tsv", 5)
    script:
        "scripts/python/time_multimodal_survival_pred.py"


rule time_time_gdp:
    output:
        "results_reproduced/timings/gdp_{cancer}",
    log:
        "logs/timings/gdp_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/gdp_{cancer}.tsv", 5)
    script:
        "scripts/python/time_gdp.py"


rule time_time_survival_net:
    output:
        "results_reproduced/timings/survival_net_{cancer}",
    log:
        "logs/timings/survival_net_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/survival_net_{cancer}.tsv", 5)
    script:
        "scripts/python/time_survival_net.py"


rule time_time_customics:
    output:
        "results_reproduced/timings/customics_{cancer}",
    log:
        "logs/timings/customics_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/customics_{cancer}.tsv", 5)
    script:
        "scripts/python/time_customics.py"


rule time_time_denoising_ae:
    output:
        "results_reproduced/timings/denoising_ae_{cancer}",
    log:
        "logs/timings/denoising_ae_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/denoising_ae_{cancer}.tsv", 5)
    script:
        "scripts/R/time_denoising_ae.R"


rule time_time_make_salmon_data:
    output:
        "results_reproduced/timings/make_salmon_data_{cancer}",
    log:
        "logs/timings/make_salmon_data_{cancer}.log",
    threads: 1
    benchmark:
        repeat("benchmarks/timings/make_salmon_data_{cancer}.tsv", 5)
    script:
        "scripts/R/time_make_salmon_data.R"
