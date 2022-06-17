import argparse
from ast import arg
import os
import fnmatch
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pandas.api.types import is_string_dtype, is_numeric_dtype
import fnmatch
from itertools import chain, combinations

base_variables = ["AGE_IN_DAYS", "OS_DAYS", "OS_STATUS"]
base_var_to_drop = ["AGE", "OS_MONTHS", "PROTOCOL", "ETHNICITY"]
pattern = ["clinical_patient", "mrna_seq_rpkm", "cna", "methylation_hm27", "mirna", "mutation"]
modalities = ["clinical", "gex", "cnv", "meth", "mirna", "mut"]


def process_clinical(file, base_variables, base_var_to_drop, dummify: bool = False):
    categorical_var = []
    numeric_var = []
    df = pd.read_csv(file, sep="\t", index_col=0, header=4)

    columns = df.columns.drop(base_variables + base_var_to_drop)
    for var in columns:

        if is_string_dtype(df[var]):

            if any(
                s in var
                for s in [
                    "KARYOTYPE",
                    "ALTERNATE_THERAPY",
                    "CYTOGENETIC_CODE",
                    "ISCN",
                    "ICDO",
                    "ICD",
                    "PERCENTAGE",
                    "PERCENT",
                ]
            ):
                continue
            categorical_var.append(var)
        elif is_numeric_dtype(df[var]) and not any(df[var].isna()):
            if any(s in var for s in ["PERCENTAGE", "PERCENT"]):
                continue
            numeric_var.append(var)

    missing_survival = df[df["OS_STATUS"].isna()].index.values
    missing_osdays = df[df["OS_days"].isna()].index.values
    zero_OS = df[df["OS_DAYS"].astype(float) == 0].index.values
    ids_to_remove = np.union1d(missing_survival, zero_OS, missing_osdays)

    df = df.drop(index=df.index[df.index.isin(ids_to_remove)])

    clin_df = df[base_variables + categorical_var + numeric_var]

    clin_df = clin_df.apply(lambda x: x.str.lower() if is_string_dtype(x) else x)

    clin_df = clin_df.astype({"OS_DAYS": np.float64, "AGE_IN_DAYS": np.float64})
    clin_df["OS_STATUS"] = clin_df["OS_STATUS"].replace({"0:living": 0, "1:deceased": 1})
    clin_df = clin_df.replace(["unknown", "not done", "not applicable", "unevaluable", "not reported"], "MISSING")
    clin_df = clin_df.fillna("MISSING")

    if dummify:
        dummy_df = pd.get_dummies(clin_df, prefix_sep="_", columns=categorical_var)
        final_clin_df = pd.concat([dummy_df, clin_df[base_variables + numeric_var]], join="inner", axis=1)
    else:
        final_clin_df = clin_df

    return final_clin_df, ids_to_remove


def process_mutation(mutation_df, save_here=None):

    mutation_df["Variant_Classification"] = [
        0 if value == "Silent" else 1 for value in mutation_df["Variant_Classification"]
    ]
    mutation_data = mutation_df.pivot_table(
        index="Tumor_Sample_Barcode",
        columns="Hugo_Symbol",
        values="Variant_Classification",
        aggfunc=sum,
    )

    if save_here:
        mutation_data.to_csv(save_here)

    return mutation_data


def process_molecular(file, ids_to_drop=None, mut_path=None, impute=False):
    df = pd.read_csv(file, sep="\t", index_col=0)
    if "mutation" in file:
        df = process_mutation(df, mut_path)
    df = df.T
    if "Entrez_Gene_Id" in df.index:
        df = df.drop(index="Entrez_Gene_Id")

    index_split = np.array(list(map(lambda x: x.rsplit("-", 1), df.index)))
    df.index = index_split[:, 0]
    df["tss"] = index_split[:, 1]
    df = df[df["tss"].isin(["01", "03", "09"])]

    df = df.drop(index=df.index[df.index.isin(ids_to_drop)], columns="tss")

    # drop features that are missing in >10% patients or have a constant value
    print("Getting constant features to drop")
    feature_variance = df.var(axis=0, skipna=True)
    features_0_var = feature_variance.index[feature_variance == 0].values.tolist()
    print("Getting features with >10% missing to drop")
    features_missing_count = df.isna().sum()
    features_missing10 = features_missing_count.index[features_missing_count > (0.1 * len(df))].values.tolist()

    df = df.drop(columns=features_0_var + features_missing10)

    # this step is v expensive in terms of runtime :/
    if impute:
        print("Median imputing missing data feature-wise")
        df = df.fillna(df.median(axis=0, skipna=True))

    if fnmatch.fnmatch(file, "*rna*"):
        print(f"Log transforming {file.split('/')[-1]}")
        df = df.applymap(lambda x: np.log2(x + 1))

    return df, features_missing_count


def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(2, len(ss) + 1)))


def get_modalities(count_dict, mod_len=3, threshold=100):
    for key in reversed(count_dict.keys()):
        value = count_dict[key]
        if value > threshold:
            modalities_to_use = key.split("^")
            if len(modalities_to_use) >= mod_len:
                break
    return modalities_to_use


parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="path to directory containing TARGET cancers")


def main(data_dir):
    cancer_folders = [folder for folder in os.listdir(data_dir) if not folder.startswith(".")]

    for cancer in cancer_folders:
        files_ = os.listdir(os.path.join(data_dir, f"{cancer}"))
        files = []
        for p in pattern:
            files.append(fnmatch.filter(files_, f"data_{p}.txt"))

        modality = [modalities[i] for i, f in enumerate(files) if f]
        files = [f[0] for f in files if f]

        save_here = os.path.join(data_dir, "processed", f"{cancer}")
        os.makedirs(save_here, exist_ok=True)
        df_dict = {}
        modality_ids = defaultdict(list)
        print(cancer, files, modality)

        for i, f in enumerate(files):

            if fnmatch.fnmatch(f, "*clinical_patient*"):

                df, ids_to_remove = process_clinical(
                    os.path.join(data_dir, f"{cancer}", f),
                    base_variables,
                    base_var_to_drop,
                    dummify=False,
                )

                len_dataset = len(df)
                modality_ids[modality[i]] = {
                    "patients": len(df.index.values),
                    "% missing": (1 - len(df.index.values) / len_dataset) * 100,
                    "patient_ids": df.index.values.tolist(),
                }

            else:
                df, features_missing_count = process_molecular(
                    os.path.join(data_dir, f"{cancer}", f), ids_to_remove, impute=True
                )

                if df.empty:
                    print(f"{modality[i]} is empty, skipping to next")
                    continue
                modality_ids[modality[i]] = {
                    "patients": len(df.index),
                    "% missing": (1 - len(df.index.values) / len_dataset) * 100,
                    "patient_ids": df.index.values.tolist(),
                }

                if not features_missing_count.sum() == 0:
                    features_missing_count.to_csv(os.path.join(save_here, f"{modality[i]}_missing_count.csv"))

            df.columns = f"{modality[i]}_" + df.columns
            df.to_csv(os.path.join(save_here, f'{f.split(".")[0]}.csv'))
            df_dict[modality[i]] = df

        print("Getting intersection lengths")
        mod_list = [set(v["patient_ids"]) for v in modality_ids.values()]
        keys = ["^".join(subset) for subset in all_subsets(modality_ids)]

        count_dict = dict.fromkeys(keys)
        for i, subset in enumerate(all_subsets(mod_list)):

            count_dict[keys[i]] = len(set.intersection(*subset))

        modality_ids.update(count_dict)
        with open(os.path.join(save_here, f"{cancer}_modality_count.json"), "w") as f:
            json.dump(modality_ids, f)

        print("Checking for modalities to use")
        modalities_to_use = get_modalities(count_dict)

        dfs_to_merge = [df_dict[key] for key in modalities_to_use]

        # THIS MERGE creates the complete df because of inner join
        print("Saving Merged DF")
        complete_df = pd.concat(dfs_to_merge, join="inner", axis=1)
        complete_df.to_csv(os.path.join(save_here, f"{cancer}_data_complete_modalities_preprocessed.csv"))

        incomplete_df = pd.concat(dfs_to_merge[1:], join="outer", axis=1)
        clinical = dfs_to_merge[0]

        incomplete_df = clinical.join(incomplete_df, how="left")
        incomplete_df = incomplete_df.rename(columns={"clinical_OS_STATUS": "OS", "clinical_OS_DAYS": "OS_days"})
        incomplete_df.index.name = "patient_id"
        incomplete_idx = np.setdiff1d(incomplete_df.index, complete_df.index)

        final_incomplete = incomplete_df.loc[incomplete_idx]
        final_incomplete.to_csv(os.path.join(save_here, f"{cancer}_data_incomplete_modalities_preprocessed.csv"))


if __name__ in "__main__":
    args = parser.parse_args()
    main(args.data_dir)
