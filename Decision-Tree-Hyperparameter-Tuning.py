from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

def prepare_dataset(df, drop_cols):
    X = df.drop(
        columns=[c for c in drop_cols if c in df.columns]
    )

    y = df["label_lapser"]

    return X, y

def get_drop_columns():
    return [
        "eff_price_mar",
        "rev_dom_mar",
        "rev_broadband_package_mar",
        "rev_broadband_mar",
        "los_x_eff_price_mar",
        "total_revenue_mar",
        "delta_eff_febmar",
        "trx_dom_mar",
        "delta_rev_dom_febmar",
        "bulan_aktif_all",
        "los_x_bulan_aktif",
        "los_x_delta_rev",
        "payload_kb_mar",
        "payload_package_mar",
        "delta_trx_dom_febmar",
        "delta_payload_febmar",
    ]

def train_baseline_decision_tree(
    X_train,
    y_train,
    random_state=42
):
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state
    )

    model.fit(X_train, y_train)

    return model

def run_gridsearch_decision_tree(
    X_train,
    y_train,
    random_state=42
):
    param_grid = {
        "max_depth": [5, 10, 15, 25, 30],
        "min_samples_split": [2, 5, 10, 15, 100],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [5, 10, 15, 20, None]
    }

    grid = GridSearchCV(
        estimator=DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid

def train_tuned_decision_tree(
    X_train,
    y_train,
    best_params,
    random_state=42
):
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=random_state,
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"]
    )

    model.fit(X_train, y_train)

    return model

def evaluate_model(
    model,
    X_test,
    y_test
):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),

        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "specificity": specificity,

        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "auc_roc": roc_auc_score(
            y_test,
            y_proba
        )
    }

    return metrics

def get_roc_curve_data(
    model,
    X_test,
    y_test
):
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_proba
    )

    auc_score = roc_auc_score(
        y_test,
        y_proba
    )

    return fpr, tpr, thresholds, auc_score

def threshold_tuning(
    y_test,
    y_proba,
    threshold_range=np.arange(0.05, 0.95, 0.01)
):
    results = []

    for th in threshold_range:
        y_pred = (y_proba >= th).astype(int)

        results.append({
            "threshold": round(float(th), 2),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "f1_score": f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        })

    return pd.DataFrame(results)

def get_best_threshold(df_threshold):
    idx = df_threshold["f1_score"].idxmax()

    return {
        "best_threshold":
            df_threshold.loc[idx, "threshold"],

        "best_f1":
            df_threshold.loc[idx, "f1_score"]
    }
