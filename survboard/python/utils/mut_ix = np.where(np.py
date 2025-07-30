mut_ix = np.where(np.isin(column_types, ["mut"]))[0]
gex_ix = np.where(np.isin(column_types, ["gex"]))[0]
cnv_ix = np.where(np.isin(column_types, ["cnv"]))[0]

genes = np.unique(
    np.unique(
        [i.rsplit("_")[1] for i in feature_names[np.concatenate((mut_ix, cnv_ix))]]
    ).tolist()
    + [i.rsplit("|")[0].rsplit("_")[1] for i in feature_names[gex_ix]]
)

gene_level_modalities = [
    (
        [i for i in feature_names[mut_ix] if i.rsplit("_")[1] == gene]
        + [i for i in feature_names[cnv_ix] if i.rsplit("_")[1] == gene]
        + [i for i in feature_names[gex_ix] if i.rsplit("|")[0].rsplit("_")[1] == gene]
    )
    for gene in genes[:10]
]
gene_level_modalities_ix = [
    [np.where(feature_name == feature_names)[0][0].tolist() for feature_name in gene]
    for gene in gene_level_modalities
]
