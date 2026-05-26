def split_features_target(df, target_col="label_lapser", drop_cols=None):
    if drop_cols is None:
        drop_cols = [
            "identifier",
            "region_lacci",
            "los_segment",
            "flag_pola_purchase",
            target_col
        ]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]

    return X, y

def apply_svmsmote(X_train, y_train, random_state=42):
    smote = SVMSMOTE(random_state=random_state)

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    X_resampled = pd.DataFrame(
        X_resampled,
        columns=X_train.columns
    )

    y_resampled = pd.Series(
        y_resampled,
        name="label_lapser"
    )

    return X_resampled, y_resampled

def get_class_distribution(y):
    values = y.value_counts().sort_index()

    return {
        "no_lapser": int(values.get(0, 0)),
        "lapser": int(values.get(1, 0))
    }

def save_smote_output(X_resampled, y_resampled, output_path):
    df_smote = X_resampled.copy()
    df_smote["label_lapser"] = y_resampled.values

    df_smote.to_excel(
        output_path,
        index=False,
        engine="openpyxl"
    )

    return df_smote
