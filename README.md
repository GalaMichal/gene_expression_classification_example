# Gene Expression Classification Example

This repository contains a Python example of binary classification using gene-expression data.

The workflow is designed as a example of molecular-signature modeling.

All data included in the repository are synthetic.

## Workflow

The example demonstrates:

- generation and loading of TPM-like gene-expression matrices
- `log2(TPM + 1)` transformation
- reference quantile normalization to harmonize expression scale across datasets
- use of a predefined candidate gene set
- nested cross-validation
- L1-regularized logistic regression
- hyperparameter optimization within training folds
- out-of-fold development predictions
- classification-threshold selection using development data only
- final model fitting on the complete development dataset
- independent validation using a locked model and threshold
- ROC analysis and confusion matrices
- export of metrics, sample-level predictions, normalized matrices and model coefficients

## Example data

They mimic a simple TPM-based gene-expression classification problem:

- genes are stored in rows
- samples are stored in columns
- a binary `response` row contains the target variable
- a predefined subset of genes is supplied in `candidate_genes.txt`
- a small subset of candidate genes carries simulated predictive signal
- the validation dataset is deliberately generated on a different absolute TPM scale

The synthetic gene names, feature selection and resulting coefficients have no biological interpretation.

## Preprocessing

The modeling pipeline uses the following sequence:

```text
TPM
  ↓
log2(TPM + 1)
  ↓
reference quantile normalization
  ↓
z-score scaling
  ↓
L1-regularized logistic regression
```

Reference quantile normalization is included to demonstrate one approach for harmonizing datasets measured on different expression scales. The reference distribution is learned from development data only.

Importantly, the quantile-normalization step is part of the scikit-learn pipeline. During nested cross-validation, it is refitted separately inside each training fold. The independent validation dataset therefore never contributes to the normalization reference used to train the model.

The validation cohort is transformed using the reference learned from the development cohort rather than being normalized jointly with development data.

## Model

The main analysis is implemented in:

```text
nested_logistic_regression.py
```

L1-regularized logistic regression is used as a sparse classifier.

For development performance, the script uses nested cross-validation. In every outer fold, preprocessing and regularization tuning are fitted using only the corresponding outer-training samples. Held-out outer-fold predictions are combined to obtain out-of-fold development probabilities.

A classification threshold is selected from these development predictions using Youden's J statistic.

The final preprocessing pipeline and classifier are then tuned and fitted using the complete development dataset. This locked pipeline and the previously selected threshold are applied without modification to the independent validation dataset.

## Outputs

The script exports:

```text
metrics.csv
model_summary.csv
development_predictions.csv
validation_predictions.csv
nested_cv_folds.csv
gene_selection_frequency.csv
signature_coefficients.csv

development_log2_qn.csv
validation_log2_qn.csv

development_roc.png
validation_roc.png
development_confusion_matrix.png
validation_confusion_matrix.png
signature_coefficients.png
normalization_comparison.png
```

Reported classification metrics include:

- ROC AUC
- accuracy
- balanced accuracy
- sensitivity
- specificity
- precision / positive predictive value
- negative predictive value
- F1 score
- Matthews correlation coefficient

`normalization_comparison.png` illustrates the deliberate scale difference between the two synthetic datasets and the effect of the preprocessing steps.

## Reproducibility

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the example:

```bash
python nested_logistic_regression.py
```

The script regenerates the synthetic TPM-like datasets using fixed random seeds and writes all results to `example_outputs/`.

## Scope

This repository is intended only as a methodological and programming example.
