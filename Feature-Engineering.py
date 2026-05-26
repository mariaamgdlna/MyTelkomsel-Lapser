def calculate_mom(current_value, previous_value):

    return np.where(
        previous_value == 0,
        0,
        (current_value - previous_value) /
        (previous_value + 1e-9)
    )


def calculate_eff_price(revenue, transaction):

    return np.where(
        transaction == 0,
        0,
        revenue / (transaction + 1e-9)
    )


def create_delta_features(df):

    df["delta_rev_dom"] = (
        df["rev_dom_feb"] -
        df["rev_dom_jan"]
    )

    df["delta_trx_dom"] = (
        df["trx_dom_feb"] -
        df["trx_dom_jan"]
    )

    df["delta_total_rev"] = (
        df["total_revenue_feb"] -
        df["total_revenue_jan"]
    )

    df["delta_payload_kb"] = (
        df["payload_kb_feb"] -
        df["payload_kb_jan"]
    )

    df["delta_payload_pkg"] = (
        df["payload_package_feb"] -
        df["payload_package_jan"]
    )

    return df


def create_mom_features(df):

    df["mom_rev_dom"] = calculate_mom(
        df["rev_dom_feb"],
        df["rev_dom_jan"]
    )

    df["mom_trx_dom"] = calculate_mom(
        df["trx_dom_feb"],
        df["trx_dom_jan"]
    )

    df["mom_total_rev"] = calculate_mom(
        df["total_revenue_feb"],
        df["total_revenue_jan"]
    )

    df["mom_payload_kb"] = calculate_mom(
        df["payload_kb_feb"],
        df["payload_kb_jan"]
    )

    return df


def create_average_features(df):

    df["avg_rev_dom"] = (
        df["rev_dom_jan"] +
        df["rev_dom_feb"]
    ) / 2

    df["avg_trx_dom"] = (
        df["trx_dom_jan"] +
        df["trx_dom_feb"]
    ) / 2

    df["avg_total_rev"] = (
        df["total_revenue_jan"] +
        df["total_revenue_feb"]
    ) / 2

    return df


def create_broadband_share_features(df):

    df["bb_share_jan"] = np.where(
        df["total_revenue_jan"] == 0,
        0,
        df["rev_broadband_package_jan"] /
        (df["total_revenue_jan"] + 1e-9)
    )

    df["bb_share_feb"] = np.where(
        df["total_revenue_feb"] == 0,
        0,
        df["rev_broadband_package_feb"] /
        (df["total_revenue_feb"] + 1e-9)
    )

    df["delta_bb_share"] = (
        df["bb_share_feb"] -
        df["bb_share_jan"]
    )

    return df


def create_activity_features(df):

    df["bulan_aktif_janfeb"] = (
        (df["flag_jan"] == "Y").astype(int) +
        (df["flag_feb"] == "Y").astype(int)
    )

    df["bulan_aktif_all"] = (
        df["bulan_aktif_janfeb"] +
        (df["flag_mar"] == "Y").astype(int)
    )

    return df


def create_eff_price_features(df):

    df["eff_price_jan"] = calculate_eff_price(
        df["rev_dom_jan"],
        df["trx_dom_jan"]
    )

    df["eff_price_feb"] = calculate_eff_price(
        df["rev_dom_feb"],
        df["trx_dom_feb"]
    )

    df["eff_price_mar"] = calculate_eff_price(
        df["rev_dom_mar"],
        df["trx_dom_mar"]
    )

    df["delta_eff_price"] = (
        df["eff_price_feb"] -
        df["eff_price_jan"]
    )

    return df


def create_los_features(df):

    df["los_group"] = pd.cut(
        df["los_encoded"],
        bins=[0, 2, 4, 7],
        labels=[1, 2, 3]
    ).astype(float).fillna(0).astype(int)

    df["los_x_bulan_aktif"] = (
        df["los_encoded"] *
        df["bulan_aktif_all"]
    )

    df["los_x_eff_price_mar"] = (
        df["los_encoded"] *
        df["eff_price_mar"]
    )

    return df


def create_business_flags(df):

    df["loyal_tapi_turun"] = (
        (df["los_encoded"] >= 5) &
        (df["delta_rev_dom"] < 0)
    ).astype(int)

    df["baru_tidak_konsisten"] = (
        (df["los_encoded"] <= 2) &
        (df["bulan_aktif_all"] <= 1)
    ).astype(int)

    return df


def get_feature_columns():

    return [

        "rev_dom_jan",
        "trx_dom_jan",
        "total_revenue_jan",

        "rev_dom_feb",
        "trx_dom_feb",
        "total_revenue_feb",

        "rev_dom_mar",
        "trx_dom_mar",
        "total_revenue_mar",

        "delta_rev_dom",
        "delta_trx_dom",
        "delta_total_rev",

        "mom_rev_dom",
        "mom_trx_dom",
        "mom_total_rev",

        "avg_rev_dom",
        "avg_trx_dom",
        "avg_total_rev",

        "bb_share_jan",
        "bb_share_feb",
        "delta_bb_share",

        "bulan_aktif_janfeb",
        "bulan_aktif_all",

        "eff_price_jan",
        "eff_price_feb",
        "eff_price_mar",
        "delta_eff_price",

        "los_encoded",
        "los_group",

        "los_x_bulan_aktif",
        "los_x_eff_price_mar",

        "loyal_tapi_turun",
        "baru_tidak_konsisten"
    ]


def prepare_model_data(df):

    feature_cols = [
        col for col in get_feature_columns()
        if col in df.columns
    ]

    X = df[feature_cols].fillna(0).values
    y = df["label_lapser"].values

    return X, y, feature_cols
