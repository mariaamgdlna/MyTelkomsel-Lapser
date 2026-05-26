import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy as pd
import pandas as pd
warnings.filterwarnings("ignore")


def load_data(file_path):
    return pd.read_excel(file_path)


def drop_invalid_rows(df):
    df = df[df["flag_pola_purchase"].notna()].copy()
    df = df[df["region_lacci"].str.strip().str.upper() != "00.UNKNOWN"].copy()
    return df


def impute_los_segment(df):
    mode_per_region = (
        df[df["los_segment"].notna()]
        .groupby("region_lacci")["los_segment"]
        .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan)
    )

    mask = df["los_segment"].isna()

    df.loc[mask, "los_segment"] = (
        df.loc[mask, "region_lacci"]
        .map(mode_per_region)
    )

    return df


def parse_purchase_flags(df):
    for i, month in enumerate(["jan", "feb", "mar"]):
        df[f"flag_{month}"] = (
            df["flag_pola_purchase"]
            .str[i]
            .str.upper()
        )

    return df


def convert_numeric_columns(df, numeric_cols):
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def clip_negative_values(df):
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in ["int64", "float64"]
    ]

    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)

    return df


def detect_duplicates(df):
    duplicate_ids = (
        df[df.duplicated(
            subset=["identifier"],
            keep=False
        )]["identifier"]
        .unique()
    )

    return duplicate_ids


def apply_iqr_capping(df, numeric_cols):
    df_capped = df.copy()

    for col in numeric_cols:

        if col not in df_capped.columns:
            continue

        q1 = df_capped[col].quantile(0.25)
        q3 = df_capped[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df_capped[col] = df_capped[col].clip(
            lower=lower,
            upper=upper
        )

    return df_capped


def save_data(df, output_path):
    df.to_excel(
        output_path,
        index=False,
        engine="openpyxl"
    )
