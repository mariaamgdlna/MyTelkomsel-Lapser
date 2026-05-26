def train_catboost_baseline(
    X_train,
    y_train,
    X_test,
    y_test,
    iterations=600,
    random_state=42
):
    train_pool = Pool(X_train, y_train)
    test_pool  = Pool(X_test, y_test)

    model = CatBoostClassifier(
        iterations=iterations,
        random_seed=random_state,
        verbose=0,
        auto_class_weights="Balanced",
        early_stopping_rounds=50
    )

    model.fit(train_pool, eval_set=test_pool)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def tune_catboost(
    X_train,
    y_train,
    param_grid,
    iterations=600,
    random_state=42
):
    train_pool = Pool(X_train, y_train)

    model = CatBoostClassifier(
        iterations=iterations,
        random_seed=random_state,
        verbose=0,
        auto_class_weights="Balanced",
        early_stopping_rounds=50
    )

    grid_result = model.grid_search(
        param_grid,
        train_pool,
        verbose=10
    )

    return grid_result["params"]

def train_catboost_tuned(
    X_train,
    y_train,
    X_test,
    y_test,
    best_params,
    iterations=600,
    random_state=42
):
    train_pool = Pool(X_train, y_train)
    test_pool  = Pool(X_test, y_test)

    model = CatBoostClassifier(
        iterations=iterations,
        random_seed=random_state,
        verbose=0,
        auto_class_weights="Balanced",
        early_stopping_rounds=50,
        **best_params
    )

    model.fit(train_pool, eval_set=test_pool)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def evaluate_model(model, X_train, y_test, y_pred, y_prob):
    auc_score = roc_auc_score(y_test, y_prob)
    specificity = calculate_specificity(y_test, y_pred)

    metrics = {
        "train_accuracy": model.score(X_train, model.predict(X_train)),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": auc_score,
        "conf_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics

def plot_confusion_and_roc(
    y_test,
    pred_base,
    pred_tuned,
    prob_base,
    prob_tuned
):
    fpr_base, tpr_base, _ = roc_curve(y_test, prob_base)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, prob_tuned)

    auc_base  = roc_auc_score(y_test, prob_base)
    auc_tuned = roc_auc_score(y_test, prob_tuned)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred_tuned),
        display_labels=["No Lapser", "Lapser"]
    ).plot(ax=axes[0], cmap="Blues")

    axes[0].set_title("Confusion Matrix — CatBoost Tuned")

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

    axes[1].plot([0, 1], [0, 1], "r--", lw=1)

    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve — CatBoost")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=15):
    fi = pd.Series(
        model.get_feature_importance(),
        index=feature_names
    ).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))

    fi[:top_n].sort_values().plot(kind="barh")

    plt.title(f"Top {top_n} Feature Importance — CatBoost")
    plt.xlabel("Importance")

    plt.tight_layout()
    plt.show()

    return fi

def compare_models(y_test, pred_base, pred_tuned, prob_base, prob_tuned):
    spec_base  = calculate_specificity(y_test, pred_base)
    spec_tuned = calculate_specificity(y_test, pred_tuned)

    comparison = {
        "accuracy_base": accuracy_score(y_test, pred_base),
        "accuracy_tuned": accuracy_score(y_test, pred_tuned),

        "precision_base": precision_score(y_test, pred_base, zero_division=0),
        "precision_tuned": precision_score(y_test, pred_tuned, zero_division=0),

        "recall_base": recall_score(y_test, pred_base, zero_division=0),
        "recall_tuned": recall_score(y_test, pred_tuned, zero_division=0),

        "f1_base": f1_score(y_test, pred_base, zero_division=0),
        "f1_tuned": f1_score(y_test, pred_tuned, zero_division=0),

        "specificity_base": spec_base,
        "specificity_tuned": spec_tuned,

        "auc_base": roc_auc_score(y_test, prob_base),
        "auc_tuned": roc_auc_score(y_test, prob_tuned),
    }

    return comparison
