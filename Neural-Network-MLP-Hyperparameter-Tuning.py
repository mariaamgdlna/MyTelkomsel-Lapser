def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def build_mlp_baseline(random_state=42):
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        batch_size=64,
        max_iter=300,
        early_stopping=True,
        random_state=random_state
    )

    return model

def train_mlp_baseline(
    X_train,
    y_train,
    X_test
):
    model = build_mlp_baseline()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def tune_mlp(
    X_train,
    y_train,
    random_state=42
):
    param_grid = {
        "hidden_layer_sizes": [
            (64,),
            (64, 32),
            (128, 64)
        ],
        "learning_rate_init": [
            0.001,
            0.0005
        ],
        "alpha": [
            0.0001,
            0.001
        ],
        "batch_size": [
            32,
            64
        ]
    }

    grid = GridSearchCV(
        estimator=MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=random_state
        ),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_params_, grid.best_score_

def train_mlp_tuned(
    X_train,
    y_train,
    X_test,
    best_params,
    random_state=42
):
    model = MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        random_state=random_state,
        **best_params
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return model, pred, prob

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def evaluate_model(
    model,
    X_train,
    y_train,
    y_test,
    y_pred,
    y_prob
):
    specificity = calculate_specificity(y_test, y_pred)

    metrics = {
        "train_accuracy": model.score(X_train, y_train),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_prob),
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

    axes[0].set_title("Confusion Matrix — Neural Network Tuned")

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
    axes[1].set_title("ROC Curve — Neural Network")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

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
            precision_score(y_test, pred_base, zero_division=0),

        "precision_tuned":
            precision_score(y_test, pred_tuned, zero_division=0),

        "recall_base":
            recall_score(y_test, pred_base, zero_division=0),

        "recall_tuned":
            recall_score(y_test, pred_tuned, zero_division=0),

        "f1_base":
            f1_score(y_test, pred_base, zero_division=0),

        "f1_tuned":
            f1_score(y_test, pred_tuned, zero_division=0),

        "auc_base":
            roc_auc_score(y_test, prob_base),

        "auc_tuned":
            roc_auc_score(y_test, prob_tuned),
    }

    return comparison

def threshold_tuning(
    y_test,
    y_prob,
    start=0.05,
    stop=0.95,
    step=0.01
):
    thresholds = np.arange(start, stop, step)

    f1_scores = []
    recall_scores = []
    precision_scores = []

    for th in thresholds:
        pred = (y_prob >= th).astype(int)

        f1_scores.append(
            f1_score(y_test, pred, zero_division=0)
        )

        recall_scores.append(
            recall_score(y_test, pred, zero_division=0)
        )

        precision_scores.append(
            precision_score(y_test, pred, zero_division=0)
        )

    best_idx = np.argmax(f1_scores)

    result = {
        "best_threshold": round(float(thresholds[best_idx]), 2),
        "best_f1": max(f1_scores),
        "thresholds": thresholds,
        "f1_scores": f1_scores,
        "recall_scores": recall_scores,
        "precision_scores": precision_scores
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

    plt.title("Threshold Tuning — Neural Network")

    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
