import json
import os

import numpy as np
import pandas as pd

cancer_list = []
project_list = []
n_modalities_list = []
n_clinical_list = []
n_gex_list = []
n_mut_list = []
n_meth_list = []
n_cnv_list = []
n_rppa_list = []
n_mirna_list = []
p_list = []
n_list = []
e_list = []
n_incomplete_list = []
e_incomplete_list = []

with open(os.path.join("./config/", "config.json"), "r") as f:
    config = json.load(f)
config.get("random_state")

for project in ["METABRIC", "TCGA", "ICGC", "TARGET"]:
    for cancer in config[f"{project.lower()}_cancers"]:
        cancer_list.append(cancer)
        project_list.append(project)
        df_complete = pd.read_csv(
            f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
        )
        df_incomplete = pd.read_csv(
            f"./data_reproduced/{project}/{cancer}_data_incomplete_modalities_preprocessed.csv"
        )
        modalities = [i.rsplit("_")[0] for i in df_complete.columns]
        n_modalities_list.append(np.unique(modalities).shape[0] - 2)
        n_clinical_list.append(modalities.count("clinical"))
        n_gex_list.append(modalities.count("gex"))
        n_mut_list.append(modalities.count("mut"))
        n_meth_list.append(modalities.count("meth"))
        n_cnv_list.append(modalities.count("cnv"))
        n_rppa_list.append(modalities.count("rppa"))
        n_mirna_list.append(modalities.count("mirna"))
        p_list.append(df_complete.shape[1])
        n_list.append(df_complete.shape[0])
        e_list.append(np.sum(df_complete.OS))

        n_incomplete_list.append(df_incomplete.shape[0])
        e_incomplete_list.append(np.sum(df_incomplete.OS))

pd.DataFrame(
    {
        "cancer": cancer_list,
        "project": project_list,
        "n_modalities": n_modalities_list,
        "n_clinical": n_clinical_list,
        "n_gex": n_gex_list,
        "n_mut": n_mut_list,
        "n_meth": n_meth_list,
        "n_cnv": n_cnv_list,
        "n_rppa": n_rppa_list,
        "n_mirna": n_mirna_list,
        "p": p_list,
        "n": n_list,
        "e": e_list,
        "n_incomplete": n_incomplete_list,
        "e_incomplete_list": e_incomplete_list,
    }
).to_csv("./tables_reproduced/survboard_final_table_S2.csv", index=False)
