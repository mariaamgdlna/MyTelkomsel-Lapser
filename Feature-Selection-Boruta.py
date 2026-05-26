from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def initialize_random_forest():

    return RandomForestClassifier(
        n_jobs=1,
        class_weight="balanced",
        max_depth=5,
        random_state=42
    )

def initialize_boruta(random_forest_model):

    return BorutaPy(
        estimator=random_forest_model,
        n_estimators="auto",
        perc=100,
        alpha=0.05,
        max_iter=100,
        random_state=42,
        verbose=0
    )

def run_boruta_feature_selection(X, y):

    rf_model = initialize_random_forest()

    boruta_model = initialize_boruta(rf_model)

    boruta_model.fit(X, y)

    return boruta_model

def create_boruta_results(
    boruta_model,
    feature_columns
):

    results = []

    for idx, feature in enumerate(feature_columns):

        rank = boruta_model.ranking_[idx]

        if boruta_model.support_[idx]:
            status = "Confirmed"

        elif boruta_model.support_weak_[idx]:
            status = "Tentative"

        else:
            status = "Rejected"

        results.append({
            "feature": feature,
            "rank": rank,
            "status": status
        })

    results_df = pd.DataFrame(results)

    return results_df.sort_values("rank")

def get_selected_features(results_df):

    confirmed_features = (
        results_df[
            results_df["status"] == "Confirmed"
        ]["feature"]
        .tolist()
    )

    tentative_features = (
        results_df[
            results_df["status"] == "Tentative"
        ]["feature"]
        .tolist()
    )

    final_features = (
        confirmed_features +
        tentative_features
    )

    if "los_encoded" not in final_features:
        final_features.append("los_encoded")

    return final_features

def save_selected_features(
    df,
    selected_features,
    output_path
):

    save_columns = (
        selected_features +
        [
            "identifier",
            "region_lacci",
            "los_segment",
            "label_lapser"
        ]
    )

    save_columns = [
        col for col in save_columns
        if col in df.columns
    ]

    df[save_columns].to_excel(
        output_path,
        index=False,
        engine="openpyxl"
    )

boruta_model = run_boruta_feature_selection(
    X,
    y
)

boruta_results = create_boruta_results(
    boruta_model,
    feature_cols
)

selected_features = get_selected_features(
    boruta_results
)
