from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    validation_curve
)
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

def build_random_forest_baseline(random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
        class_weight="balanced"
    )

    return model

def train_random_forest_baseline(
    X_train,
    y_train,
    X_test
):
    model = build_random_forest_baseline()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def randomized_search_rf(
    X_train,
    y_train,
    random_state=42
):
    random_grid = {
        "n_estimators": [
            100,
            200,
            300,
            500
        ],

        "max_depth": [
            5,
            10,
            15,
            20,
            None
        ],

        "max_features": [
            "sqrt",
            "log2",
            None
        ],

        "min_samples_split": [
            2,
            5,
            10
        ],

        "min_samples_leaf": [
            1,
            2,
            5
        ],

        "bootstrap": [
            True,
            False
        ]
    }

    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=random_state,
            n_jobs=1
        ),

        param_distributions=random_grid,

        n_iter=20,
        scoring="f1",
        cv=3,
        verbose=1,
        random_state=random_state,
        n_jobs=-1,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)

    return (
        random_search.best_params_,
        random_search.best_score_,
        random_search.best_estimator_
    )

def grid_search_rf(
    X_train,
    y_train,
    best_random_params,
    random_state=42
):
    bp = best_random_params

    param_grid = {
        "n_estimators": [
            max(50, bp["n_estimators"] - 100),
            bp["n_estimators"],
            bp["n_estimators"] + 100
        ],

        "max_depth": (
            [bp["max_depth"]]
            if bp["max_depth"] is None
            else [
                max(3, bp["max_depth"] - 5),
                bp["max_depth"],
                bp["max_depth"] + 5
            ]
        ),

        "max_features": [
            bp["max_features"]
        ],

        "min_samples_split": [
            max(2, bp["min_samples_split"] - 2),
            bp["min_samples_split"],
            bp["min_samples_split"] + 2
        ],

        "min_samples_leaf": [
            max(1, bp["min_samples_leaf"] - 1),
            bp["min_samples_leaf"],
            bp["min_samples_leaf"] + 1
        ],

        "bootstrap": [
            bp["bootstrap"]
        ]
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=random_state,
            n_jobs=1
        ),

        param_grid=param_grid,

        scoring="f1",
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_params_,
        grid_search.best_score_,
        grid_search.best_estimator_
    )

def train_random_forest_tuned(
    X_train,
    y_train,
    X_test,
    best_params,
    random_state=42
):
    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=1,
        **best_params
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp)

def evaluate_model(
    model,
    X_train,
    y_train,
    y_test,
    y_pred,
    y_prob
):
    specificity = calculate_specificity(
        y_test,
        y_pred
    )

    metrics = {
        "train_accuracy":
            model.score(X_train, y_train),

        "accuracy":
            accuracy_score(y_test, y_pred),

        "precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "specificity":
            specificity,

        "f1_score":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "auc_roc":
            roc_auc_score(y_test, y_prob),

        "conf_matrix":
            confusion_matrix(y_test, y_pred)
    }

    return metrics

def plot_confusion_and_roc(
    y_test,
    pred_base,
    pred_tuned,
    prob_base,
    prob_tuned
):
    fpr_base, tpr_base, _ = roc_curve(
        y_test,
        prob_base
    )

    fpr_tuned, tpr_tuned, _ = roc_curve(
        y_test,
        prob_tuned
    )

    auc_base = roc_auc_score(
        y_test,
        prob_base
    )

    auc_tuned = roc_auc_score(
        y_test,
        prob_tuned
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4)
    )

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred_tuned),
        display_labels=["No Lapser", "Lapser"]
    ).plot(
        ax=axes[0],
        cmap="Blues"
    )

    axes[0].set_title(
        "Confusion Matrix — Random Forest Tuned"
    )

    axes[1].plot(
        fpr_base,
        tpr_base,
        lw=2,
        linestyle="--",
        label=f"Baseline AUC={auc_base:.4f}"
    )

    axes[1].plot(
        fpr_tuned,
        tpr_tuned,
        lw=2,
        label=f"Tuned AUC={auc_tuned:.4f}"
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        "r--",
        lw=1
    )

    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")

    axes[1].set_title(
        "ROC Curve — Random Forest"
    )

    axes[1].legend()

    plt.tight_layout()
    plt.show()

def feature_importance_rf(
    model,
    feature_names,
    top_n=15
):
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    importance[:top_n].sort_values().plot(
        kind="barh"
    )

    plt.title(
        f"Top {top_n} Feature Importance — Random Forest"
    )

    plt.xlabel("Importance")

    plt.tight_layout()
    plt.show()

    return importance

def compare_models(
    y_test,
    pred_base,
    pred_tuned,
    prob_base,
    prob_tuned
):
    comparison = {
        "accuracy_base":
            accuracy_score(y_test, pred_base),

        "accuracy_tuned":
            accuracy_score(y_test, pred_tuned),

        "precision_base":
            precision_score(
                y_test,
                pred_base,
                zero_division=0
            ),

        "precision_tuned":
            precision_score(
                y_test,
                pred_tuned,
                zero_division=0
            ),

        "recall_base":
            recall_score(
                y_test,
                pred_base,
                zero_division=0
            ),

        "recall_tuned":
            recall_score(
                y_test,
                pred_tuned,
                zero_division=0
            ),

        "f1_base":
            f1_score(
                y_test,
                pred_base,
                zero_division=0
            ),

        "f1_tuned":
            f1_score(
                y_test,
                pred_tuned,
                zero_division=0
            ),

        "auc_base":
            roc_auc_score(
                y_test,
                prob_base
            ),

        "auc_tuned":
            roc_auc_score(
                y_test,
                prob_tuned
            )
    }

    return comparison

def threshold_tuning(
    y_test,
    y_prob,
    start=0.05,
    stop=0.95,
    step=0.01
):
    thresholds = np.arange(
        start,
        stop,
        step
    )

    f1_scores = []
    recall_scores = []
    precision_scores = []

    for th in thresholds:
        pred = (y_prob >= th).astype(int)

        f1_scores.append(
            f1_score(
                y_test,
                pred,
                zero_division=0
            )
        )

        recall_scores.append(
            recall_score(
                y_test,
                pred,
                zero_division=0
            )
        )

        precision_scores.append(
            precision_score(
                y_test,
                pred,
                zero_division=0
            )
        )

    best_idx = np.argmax(f1_scores)

    result = {
        "best_threshold":
            round(float(thresholds[best_idx]), 2),

        "best_f1":
            max(f1_scores),

        "thresholds":
            thresholds,

        "f1_scores":
            f1_scores,

        "recall_scores":
            recall_scores,

        "precision_scores":
            precision_scores
    }

    return result

def plot_threshold_tuning(
    thresholds,
    f1_scores,
    recall_scores,
    precision_scores,
    best_threshold
):
    plt.figure(figsize=(11, 4))

    plt.plot(
        thresholds,
        f1_scores,
        lw=2,
        label="F1 Score"
    )

    plt.plot(
        thresholds,
        recall_scores,
        lw=2,
        label="Recall"
    )

    plt.plot(
        thresholds,
        precision_scores,
        lw=2,
        label="Precision"
    )

    plt.axvline(
        0.5,
        linestyle=":",
        lw=1.5,
        label="Default th=0.50"
    )

    plt.axvline(
        best_threshold,
        linestyle="--",
        lw=2,
        label=f"Optimal th={best_threshold}"
    )

    plt.xlabel("Threshold")
    plt.ylabel("Score")

    plt.title(
        "Threshold Tuning — Random Forest"
    )

    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
