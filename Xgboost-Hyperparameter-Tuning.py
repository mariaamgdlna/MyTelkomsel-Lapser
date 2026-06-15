import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import randint, uniform

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

def train_xgboost_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=300,
    random_state=42
):
    """
    Train baseline XGBoost model.
    """

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob
  
def tune_xgboost(
    X_train,
    y_train,
    random_state=42,
    n_iter=30
):
    """
    Hyperparameter tuning using RandomizedSearchCV.
    """

    param_dist = {
        "max_depth": randint(3, 11),

        "min_child_weight": randint(1, 10),

        "subsample": uniform(0.6, 0.4),

        "colsample_bytree": uniform(0.6, 0.4),

        "gamma": uniform(0.0, 5.0),

        "reg_alpha": [
            1e-8,
            1e-5,
            1e-3,
            0.01,
            0.1,
            1.0
        ],

        "reg_lambda": [
            0.01,
            0.1,
            1,
            5,
            10
        ],

        "learning_rate": uniform(0.01, 0.29),

        "n_estimators": randint(100, 500)
    }

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=random_state
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    for k, v in search.best_params_.items():
        print(f"{k}: {v}")

    print(f"\nBest CV F1 Score: {search.best_score_:.4f}")

    return search.best_params_

def train_xgboost_tuned(
    X_train,
    y_train,
    X_test,
    y_test,
    best_params,
    random_state=42
):
    """
    Train tuned XGBoost model using best hyperparameters.
    """

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        **best_params
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def calculate_specificity(
    y_true,
    y_pred
):
    """
    Calculate specificity score.
    """

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
    """
    Evaluate classification performance.
    """

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    specificity = calculate_specificity(
        y_test,
        y_pred
    )

    metrics = {
        "train_accuracy":
            accuracy_score(
                y_train,
                model.predict(X_train)
            ),

        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

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
            auc_score,

        "conf_matrix":
            confusion_matrix(
                y_test,
                y_pred
            )
    }

    return metrics

def plot_confusion_and_roc(
    y_test,
    pred_base,
    pred_tuned,
    prob_base,
    prob_tuned
):
    """
    Plot confusion matrix and ROC curve.
    """

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
        figsize=(13, 5)
    )

    ConfusionMatrixDisplay(
        confusion_matrix(
            y_test,
            pred_tuned
        ),
        display_labels=[
            "No Churn",
            "Churn"
        ]
    ).plot(
        ax=axes[0],
        cmap="Blues"
    )

    axes[0].set_title(
        "Confusion Matrix - XGBoost Tuned"
    )

    axes[1].plot(
        fpr_base,
        tpr_base,
        linestyle="--",
        linewidth=2,
        label=f"Baseline AUC = {auc_base:.4f}"
    )

    axes[1].plot(
        fpr_tuned,
        tpr_tuned,
        linewidth=2,
        label=f"Tuned AUC = {auc_tuned:.4f}"
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        "r--",
        linewidth=1
    )

    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve - XGBoost")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_feature_importance(
    model,
    feature_names,
    top_n=15
):
    """
    Plot feature importance.
    """

    fi = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(
        ascending=False
    )

    plt.figure(
        figsize=(10, 6)
    )

    fi[:top_n] \
        .sort_values() \
        .plot(
            kind="barh"
        )

    plt.title(
        f"Top {top_n} Feature Importance - XGBoost"
    )

    plt.xlabel(
        "Importance Score"
    )

    plt.tight_layout()
    plt.show()

    return fi

def compare_models(
    y_test,
    pred_base,
    pred_tuned,
    prob_base,
    prob_tuned
):
    """
    Compare baseline and tuned model.
    """

    comparison = {
        "accuracy_base":
            accuracy_score(
                y_test,
                pred_base
            ),

        "accuracy_tuned":
            accuracy_score(
                y_test,
                pred_tuned
            ),

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

        "specificity_base":
            calculate_specificity(
                y_test,
                pred_base
            ),

        "specificity_tuned":
            calculate_specificity(
                y_test,
                pred_tuned
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
