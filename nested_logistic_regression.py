#!/usr/bin/env python
"""
Generic gene-expression classification example.

This script demonstrates:
1. generation/loading of synthetic TPM-like development and validation datasets,
2. log2(TPM + 1) transformation,
3. reference quantile normalization fitted on training data only,
4. use of a predefined candidate gene set,
5. nested cross-validation for unbiased development predictions,
6. L1-regularized logistic regression,
7. threshold selection from development predictions only,
8. independent validation with the locked model and threshold,
9. export of metrics, predictions, figures, normalized matrices, and coefficients.

All data are synthetic. The resulting signature has no biological or clinical meaning.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42

N_DEVELOPMENT = 300
N_VALIDATION = 150
N_GENES = 80
N_CANDIDATE_GENES = 40

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 5

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "example_outputs"

DEVELOPMENT_FILE = DATA_DIR / "development_data.csv"
VALIDATION_FILE = DATA_DIR / "validation_data.csv"
CANDIDATE_FILE = DATA_DIR / "candidate_genes.txt"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# =============================================================================
# PREPROCESSING
# =============================================================================

class Log2TPMTransformer(BaseEstimator, TransformerMixin):
    """Apply log2(TPM + 1) while rejecting negative input values."""

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        if np.any(X_array < 0):
            raise ValueError("TPM values must be non-negative.")
        self.n_features_in_ = X_array.shape[1]
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        if np.any(X_array < 0):
            raise ValueError("TPM values must be non-negative.")
        return np.log2(X_array + 1.0)


class ReferenceQuantileNormalizer(BaseEstimator, TransformerMixin):
    """
    Reference quantile normalization for samples x genes data.

    During fit, a reference distribution is learned from TRAINING samples only:
    each sample's feature values are sorted and the mean value at every rank is
    stored. During transform, each sample is independently mapped onto that
    reference distribution according to its within-sample feature ranks.

    Because this transformer is inside the sklearn Pipeline, nested CV fits the
    reference separately within every training fold. The external validation
    cohort never contributes to the reference distribution.
    """

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)

        if X_array.ndim != 2:
            raise ValueError("Expected a 2D samples x genes matrix.")

        self.n_features_in_ = X_array.shape[1]

        # Sort genes within each training sample; average across samples at each rank.
        sorted_values = np.sort(X_array, axis=1)
        self.reference_quantiles_ = np.mean(sorted_values, axis=0)

        return self

    @staticmethod
    def _average_ranks(values):
        """Return zero-based average ranks, including tied values."""
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)

        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and sorted_values[end] == sorted_values[start]:
                end += 1

            average_rank = (start + end - 1) / 2.0
            ranks[order[start:end]] = average_rank
            start = end

        return ranks

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)

        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("Feature count differs from the fitted reference.")

        reference_positions = np.arange(self.n_features_in_, dtype=float)
        normalized = np.empty_like(X_array, dtype=float)

        for i, sample in enumerate(X_array):
            ranks = self._average_ranks(sample)
            normalized[i] = np.interp(
                ranks,
                reference_positions,
                self.reference_quantiles_,
            )

        return normalized


# =============================================================================
# SYNTHETIC TPM-LIKE DATA
# =============================================================================

def generate_synthetic_dataset(
    n_samples,
    sample_prefix,
    seed,
    tpm_scale_factor=1.0,
):
    """
    Generate a synthetic TPM-like gene-expression matrix.

    Rows are genes and columns are samples. A final row named "response"
    contains the binary target.

    The validation cohort can be generated on a different overall TPM scale to
    illustrate why scale-aware preprocessing can be useful.
    """
    rng = np.random.default_rng(seed)

    gene_names = [f"GENE_{i:03d}" for i in range(1, N_GENES + 1)]
    sample_names = [f"{sample_prefix}_{i:03d}" for i in range(1, n_samples + 1)]

    # Latent log2-expression values with sample-to-sample correlation structure.
    latent_1 = rng.normal(0, 1, n_samples)
    latent_2 = rng.normal(0, 1, n_samples)

    log_expression = rng.normal(5.0, 1.1, size=(n_samples, N_GENES))
    log_expression[:, :15] += 0.45 * latent_1[:, None]
    log_expression[:, 15:30] += 0.35 * latent_2[:, None]

    # Define outcome from a small subset of candidate genes.
    informative_idx = np.array([1, 4, 8, 12, 18, 24, 31, 36])
    centered_signal = log_expression[:, informative_idx] - 5.0
    beta = np.array([0.85, -0.75, 0.70, -0.65, 0.60, 0.55, -0.50, 0.45])

    linear_score = centered_signal @ beta
    linear_score += rng.normal(0, 1.8, n_samples)

    probability = 1.0 / (1.0 + np.exp(-linear_score))
    y = rng.binomial(1, probability)

    # Convert latent log-expression to non-negative TPM-like values.
    tpm = np.maximum(0.0, np.power(2.0, log_expression) - 1.0)

    # Deliberately place the independent cohort on a different absolute scale.
    tpm *= tpm_scale_factor

    matrix = pd.DataFrame(tpm.T, index=gene_names, columns=sample_names)
    matrix.loc["response"] = y

    return matrix


def ensure_example_data():
    """Create deterministic synthetic development/validation datasets."""
    development = generate_synthetic_dataset(
        n_samples=N_DEVELOPMENT,
        sample_prefix="DEV",
        seed=RANDOM_STATE,
        tpm_scale_factor=1.0,
    )

    validation = generate_synthetic_dataset(
        n_samples=N_VALIDATION,
        sample_prefix="VAL",
        seed=RANDOM_STATE + 1,
        tpm_scale_factor=3.0,
    )

    development.to_csv(DEVELOPMENT_FILE)
    validation.to_csv(VALIDATION_FILE)

    candidate_genes = [f"GENE_{i:03d}" for i in range(1, N_CANDIDATE_GENES + 1)]
    CANDIDATE_FILE.write_text("\n".join(candidate_genes) + "\n", encoding="utf-8")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_expression_matrix(path):
    """Load a genes-in-rows, samples-in-columns matrix."""
    return pd.read_csv(path, index_col=0)


def load_candidate_genes(path):
    """Load one predefined candidate gene per line."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def extract_X_y(matrix, candidate_genes):
    """
    Convert the matrix into sklearn format:
    X = samples x genes
    y = binary response
    """
    if "response" not in matrix.index:
        raise KeyError("Expected a row named 'response'.")

    common_genes = [gene for gene in candidate_genes if gene in matrix.index]
    if len(common_genes) < 2:
        raise ValueError("Too few candidate genes are present in the dataset.")

    X = matrix.loc[common_genes].T.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(matrix.loc["response"], errors="raise").astype(int)

    if X.isna().any().any():
        raise ValueError("The example expects complete TPM values.")

    if set(y.unique()) - {0, 1}:
        raise ValueError("The response row must contain only 0 and 1.")

    return X, y


# =============================================================================
# MODEL
# =============================================================================

def make_pipeline():
    """
    Full preprocessing + sparse logistic-regression pipeline.

    Order:
        TPM
        -> log2(TPM + 1)
        -> reference quantile normalization
        -> z-score scaling
        -> L1 logistic regression

    Because preprocessing is inside the Pipeline, every learned preprocessing
    step is fitted independently inside the training folds during nested CV.
    """
    return Pipeline(
        [
            ("log2_tpm", Log2TPMTransformer()),
            ("quantile", ReferenceQuantileNormalizer()),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def nested_oof_predictions(X, y):
    """Generate unbiased out-of-fold probabilities using nested CV."""
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probability = np.zeros(len(y), dtype=float)
    selection_count = pd.Series(0, index=X.columns, dtype=int)
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_outer_train = X.iloc[train_idx]
        y_outer_train = y.iloc[train_idx]
        X_outer_test = X.iloc[test_idx]
        y_outer_test = y.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + fold,
        )

        search = GridSearchCV(
            estimator=make_pipeline(),
            param_grid={"model__C": C_GRID},
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_outer_train, y_outer_train)

        fold_probability = search.predict_proba(X_outer_test)[:, 1]
        oof_probability[test_idx] = fold_probability

        coefficients = search.best_estimator_.named_steps["model"].coef_.ravel()
        selected = X.columns[np.abs(coefficients) > 1e-12]
        selection_count.loc[selected] += 1

        fold_rows.append(
            {
                "outer_fold": fold,
                "best_C": search.best_params_["model__C"],
                "n_selected_genes": len(selected),
                "outer_auc": roc_auc_score(y_outer_test, fold_probability),
            }
        )

    selection_frequency = pd.DataFrame(
        {
            "gene": selection_count.index,
            "selected_in_outer_folds": selection_count.values,
            "selection_frequency": selection_count.values / N_OUTER_SPLITS,
        }
    ).sort_values(
        ["selection_frequency", "selected_in_outer_folds"],
        ascending=False,
    )

    return oof_probability, pd.DataFrame(fold_rows), selection_frequency


def choose_youden_threshold(y_true, probability):
    """Choose a classification threshold from development predictions only."""
    fpr, tpr, thresholds = roc_curve(y_true, probability)
    finite = np.isfinite(thresholds)
    youden = tpr[finite] - fpr[finite]
    return float(thresholds[finite][np.argmax(youden)])


def calculate_metrics(y_true, probability, threshold):
    """Calculate threshold-independent and threshold-dependent metrics."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)

    cm = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    metrics = {
        "AUC": roc_auc_score(y_true, probability),
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, prediction),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, prediction),
        "Sensitivity": recall_score(y_true, prediction, zero_division=0),
        "Specificity": specificity,
        "Precision_PPV": precision_score(y_true, prediction, zero_division=0),
        "NPV": npv,
        "F1": f1_score(y_true, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y_true, prediction),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    return metrics, prediction, cm


def fit_final_model(X, y):
    """Tune and fit the final pipeline on the complete development dataset."""
    cv = StratifiedKFold(
        n_splits=N_INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + 100,
    )

    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"model__C": C_GRID},
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X, y)
    return search.best_estimator_, search.best_params_


# =============================================================================
# EXPORT PREPROCESSED MATRICES
# =============================================================================

def transform_before_scaling(fitted_pipeline, X):
    """
    Apply log2(TPM+1) and the fitted training-reference quantile normalization.

    This is used only to export reviewable normalized matrices.
    """
    logged = fitted_pipeline.named_steps["log2_tpm"].transform(X)
    quantile_normalized = fitted_pipeline.named_steps["quantile"].transform(logged)
    return quantile_normalized


def export_preprocessed_matrix(fitted_pipeline, X, path):
    normalized = transform_before_scaling(fitted_pipeline, X)
    table = pd.DataFrame(
        normalized,
        index=X.index,
        columns=X.columns,
    )
    table.index.name = "sample_id"
    table.to_csv(path)


# =============================================================================
# FIGURES
# =============================================================================

def plot_roc(y_true, probability, title, filename):
    fpr, tpr, _ = roc_curve(y_true, probability)
    auc = roc_auc_score(y_true, probability)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_xticks([0, 1], labels=["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)

    threshold = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_coefficients(signature):
    selected = signature.loc[signature["selected"]].copy()
    if selected.empty:
        return

    selected = selected.sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(selected) + 1)))
    ax.barh(selected["gene"], selected["coefficient"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Standardized logistic-regression coefficient")
    ax.set_title("Final model coefficients")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "signature_coefficients.png", dpi=300)
    plt.close(fig)


def plot_scale_normalization_example(X_development, X_validation, fitted_pipeline):
    """
    Visualize the deliberate scale mismatch and its correction.

    We plot per-sample medians at three stages:
    raw TPM -> log2(TPM+1) -> reference quantile normalized.
    """
    dev_raw = X_development.to_numpy(dtype=float)
    val_raw = X_validation.to_numpy(dtype=float)

    log_transformer = fitted_pipeline.named_steps["log2_tpm"]
    quantile_transformer = fitted_pipeline.named_steps["quantile"]

    dev_log = log_transformer.transform(X_development)
    val_log = log_transformer.transform(X_validation)

    dev_qn = quantile_transformer.transform(dev_log)
    val_qn = quantile_transformer.transform(val_log)

    stages = [
        (
            "Raw TPM",
            np.median(dev_raw, axis=1),
            np.median(val_raw, axis=1),
        ),
        (
            "log2(TPM+1)",
            np.median(dev_log, axis=1),
            np.median(val_log, axis=1),
        ),
        (
            "Reference quantile normalized",
            np.median(dev_qn, axis=1),
            np.median(val_qn, axis=1),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (title, dev_values, val_values) in zip(axes, stages):
        ax.boxplot([dev_values, val_values], tick_labels=["Development", "Validation"])
        ax.set_title(title)
        ax.set_ylabel("Per-sample median")

    fig.suptitle("Example preprocessing across datasets")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "normalization_comparison.png", dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    ensure_example_data()

    development_matrix = load_expression_matrix(DEVELOPMENT_FILE)
    validation_matrix = load_expression_matrix(VALIDATION_FILE)
    candidate_genes = load_candidate_genes(CANDIDATE_FILE)

    X_development, y_development = extract_X_y(
        development_matrix,
        candidate_genes,
    )
    X_validation, y_validation = extract_X_y(
        validation_matrix,
        candidate_genes,
    )

    print(f"Development samples: {len(y_development)}")
    print(f"Validation samples:  {len(y_validation)}")
    print(f"Candidate genes:     {len(candidate_genes)}")
    print("Preprocessing:       log2(TPM+1) -> reference quantile normalization -> z-score")

    # -------------------------------------------------------------------------
    # 1. Development performance: nested out-of-fold predictions
    # -------------------------------------------------------------------------
    oof_probability, fold_results, selection_frequency = nested_oof_predictions(
        X_development,
        y_development,
    )

    locked_threshold = choose_youden_threshold(
        y_development,
        oof_probability,
    )

    development_metrics, development_prediction, development_cm = calculate_metrics(
        y_development,
        oof_probability,
        locked_threshold,
    )

    # -------------------------------------------------------------------------
    # 2. Final model: tune on complete development dataset
    # -------------------------------------------------------------------------
    final_model, best_params = fit_final_model(
        X_development,
        y_development,
    )

    final_coefficients = final_model.named_steps["model"].coef_.ravel()

    signature = pd.DataFrame(
        {
            "gene": X_development.columns,
            "coefficient": final_coefficients,
            "abs_coefficient": np.abs(final_coefficients),
            "selected": np.abs(final_coefficients) > 1e-12,
        }
    ).sort_values("abs_coefficient", ascending=False)

    # -------------------------------------------------------------------------
    # 3. Independent validation: locked model + locked threshold
    # -------------------------------------------------------------------------
    validation_probability = final_model.predict_proba(X_validation)[:, 1]

    validation_metrics, validation_prediction, validation_cm = calculate_metrics(
        y_validation,
        validation_probability,
        locked_threshold,
    )

    # -------------------------------------------------------------------------
    # 4. Export results
    # -------------------------------------------------------------------------
    metrics_table = pd.DataFrame(
        [
            {"dataset": "development_nested_OOF", **development_metrics},
            {"dataset": "independent_validation", **validation_metrics},
        ]
    )
    metrics_table.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": X_development.index,
            "true_class": y_development.values,
            "probability": oof_probability,
            "prediction": development_prediction,
        }
    ).to_csv(OUTPUT_DIR / "development_predictions.csv", index=False)

    pd.DataFrame(
        {
            "sample_id": X_validation.index,
            "true_class": y_validation.values,
            "probability": validation_probability,
            "prediction": validation_prediction,
        }
    ).to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    fold_results.to_csv(OUTPUT_DIR / "nested_cv_folds.csv", index=False)
    selection_frequency.to_csv(
        OUTPUT_DIR / "gene_selection_frequency.csv",
        index=False,
    )
    signature.to_csv(OUTPUT_DIR / "signature_coefficients.csv", index=False)

    export_preprocessed_matrix(
        final_model,
        X_development,
        OUTPUT_DIR / "development_log2_qn.csv",
    )
    export_preprocessed_matrix(
        final_model,
        X_validation,
        OUTPUT_DIR / "validation_log2_qn.csv",
    )

    pd.DataFrame(
        development_cm,
        index=["True_0", "True_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(OUTPUT_DIR / "development_confusion_matrix.csv")

    pd.DataFrame(
        validation_cm,
        index=["True_0", "True_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(OUTPUT_DIR / "validation_confusion_matrix.csv")

    pd.DataFrame(
        [
            {
                "best_C": best_params["model__C"],
                "locked_threshold": locked_threshold,
                "n_candidate_genes": len(candidate_genes),
                "n_final_selected_genes": int(signature["selected"].sum()),
                "preprocessing": (
                    "log2(TPM+1) + training-reference quantile normalization + z-score"
                ),
            }
        ]
    ).to_csv(OUTPUT_DIR / "model_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. Figures
    # -------------------------------------------------------------------------
    plot_roc(
        y_development,
        oof_probability,
        "Development ROC (nested OOF)",
        "development_roc.png",
    )

    plot_roc(
        y_validation,
        validation_probability,
        "Independent validation ROC",
        "validation_roc.png",
    )

    plot_confusion_matrix(
        development_cm,
        f"Development confusion matrix\nthreshold = {locked_threshold:.3f}",
        "development_confusion_matrix.png",
    )

    plot_confusion_matrix(
        validation_cm,
        f"Validation confusion matrix\nlocked threshold = {locked_threshold:.3f}",
        "validation_confusion_matrix.png",
    )

    plot_coefficients(signature)
    plot_scale_normalization_example(
        X_development,
        X_validation,
        final_model,
    )

    # -------------------------------------------------------------------------
    # 6. Console summary
    # -------------------------------------------------------------------------
    print("\nDevelopment metrics")
    print(pd.Series(development_metrics).to_string())

    print("\nValidation metrics")
    print(pd.Series(validation_metrics).to_string())

    print(f"\nBest final C: {best_params['model__C']}")
    print(f"Locked threshold: {locked_threshold:.4f}")
    print(f"Selected genes in final model: {int(signature['selected'].sum())}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
