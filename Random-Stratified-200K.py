import pandas as pd
import numpy as np
import random

SEPARATOR    = "|"
SAMPLE_SIZE  = 200_000
RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def get_region_distribution(file_path, separator="|"):
    """
    Calculate region distribution from large raw dataset.
    """
    region_counts = {}

    with open(file_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split(separator)
        region_idx = header.index("region_lacci")

        for line in file:
            values = line.strip().split(separator)

            try:
                region = values[region_idx]
                region_counts[region] = region_counts.get(region, 0) + 1
            except IndexError:
                continue

    total_rows = sum(region_counts.values())

    return header, region_counts, total_rows

def build_reservoir_sample(
    file_path,
    header,
    region_counts,
    total_rows,
    sample_size,
    separator="|",
    random_state=42,
    buffer_multiplier=3
):
    """
    Perform reservoir sampling with stratified region distribution.
    """
    random.seed(random_state)

    region_idx = header.index("region_lacci")

    regions_sorted = sorted(
        region_counts.items(),
        key=lambda x: -x[1]
    )

    stratify_values = [region for region, _ in regions_sorted]
    stratify_props  = [
        count / total_rows
        for _, count in regions_sorted
    ]

    quota_buffer = {
        region: min(
            int(sample_size * prop * buffer_multiplier),
            count
        )
        for (region, count), prop in zip(regions_sorted, stratify_props)
    }

    reservoirs = {
        region: []
        for region in stratify_values
    }

    seen_counter = {
        region: 0
        for region in stratify_values
    }

    with open(file_path, "r", encoding="utf-8") as file:
        file.readline()

        for line in file:
            values = line.strip().split(separator)

            try:
                region = values[region_idx]

                if region not in reservoirs:
                    continue

                quota = quota_buffer[region]
                seen_counter[region] += 1

                if seen_counter[region] <= quota:
                    reservoirs[region].append(values)

                else:
                    random_index = random.randint(
                        0,
                        seen_counter[region] - 1
                    )

                    if random_index < quota:
                        reservoirs[region][random_index] = values

            except IndexError:
                continue

    all_rows = []

    for region in stratify_values:
        all_rows.extend(reservoirs[region])

    df_buffer = pd.DataFrame(all_rows, columns=header)

    return df_buffer, stratify_values, stratify_props

def stratified_sampling(
    dataframe,
    stratify_column,
    stratify_values,
    stratify_proportions,
    sample_size,
    random_state=42
):
    """
    Generate final stratified sample.
    """
    sampled_df = pd.DataFrame(columns=dataframe.columns)

    for index, value in enumerate(stratify_values):

        if index == len(stratify_values) - 1:
            target_size = sample_size - len(sampled_df)

        else:
            target_size = int(
                sample_size * stratify_proportions[index]
            )

        filtered_df = dataframe[
            dataframe[stratify_column] == value
        ]

        sampled_part = filtered_df.sample(
            n=min(target_size, len(filtered_df)),
            replace=False,
            random_state=random_state
        )

        sampled_df = pd.concat(
            [sampled_df, sampled_part],
            ignore_index=True
        )

    return sampled_df

def convert_numeric_columns(dataframe, numeric_columns):
    """
    Convert selected columns into numeric datatype.
    """
    for column in numeric_columns:

        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            ).fillna(0)

    return dataframe

NUMERIC_COLUMNS = [
    "rev_dom_jan",
    "trx_dom_jan",
    "rev_dom_feb",
    "trx_dom_feb",
    "rev_dom_mar",
    "trx_dom_mar",
    "total_revenue_jan",
    "rev_broadband_jan",
    "rev_broadband_package_jan",
    "payload_kb_jan",
    "payload_package_jan",
    "total_revenue_feb",
    "rev_broadband_feb",
    "rev_broadband_package_feb",
    "payload_kb_feb",
    "payload_package_feb",
    "total_revenue_mar",
    "rev_broadband_mar",
    "rev_broadband_package_mar",
    "payload_kb_mar",
    "payload_package_mar"
]

header, region_counts, total_rows = get_region_distribution(
    FILE_PATH,
    separator=SEPARATOR
)

df_buffer, stratify_values, stratify_props = build_reservoir_sample(
    file_path       = FILE_PATH,
    header          = header,
    region_counts   = region_counts,
    total_rows      = total_rows,
    sample_size     = SAMPLE_SIZE,
    separator       = SEPARATOR,
    random_state    = RANDOM_STATE
)

df_buffer = convert_numeric_columns(
    df_buffer,
    NUMERIC_COLUMNS
)

df_sample = stratified_sampling(
    dataframe              = df_buffer,
    stratify_column        = "region_lacci",
    stratify_values        = stratify_values,
    stratify_proportions   = stratify_props,
    sample_size            = SAMPLE_SIZE,
    random_state           = RANDOM_STATE
)

df_sample.to_excel(
    OUTPUT_PATH,
    index=False,
    engine="openpyxl"
)
