import pandas as pd
import numpy as np


def load_data(file_path):
    return pd.read_excel(file_path)


def parse_purchase_flags(df):
    for i, month in enumerate(["jan", "feb", "mar"]):
        df[f"flag_{month}"] = (
            df["flag_pola_purchase"]
            .str[i]
            .str.upper()
        )

    return df


def encode_los_segment(df):

    los_order = [
        "0-1 M",
        "1-3 M",
        "3-6 M",
        "6-12 M",
        "1-2 Yr",
        "2-5 Yr",
        ">5 Yr"
    ]

    los_mapping = {
        value: idx + 1
        for idx, value in enumerate(los_order)
    }

    df["los_encoded"] = (
        df["los_segment"]
        .map(los_mapping)
        .fillna(0)
        .astype(int)
    )

    return df


def create_lapser_label(df):

    df["label_lapser"] = (
        df["flag_mar"] == "N"
    ).astype(int)

    return df


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


# =========================
# MAIN PROCESS
# =========================

INPUT_PATH = "Output_New_Sample_200k_Clean.xlsx"

df = load_data(INPUT_PATH)

df = parse_purchase_flags(df)

df = encode_los_segment(df)

df = create_lapser_label(df)
