# ChurnIQ — Customer Churn Intelligence

A deployment-ready Streamlit web application built around the supplied e-commerce churn project and its saved LightGBM model. It turns the existing notebook workflow into a polished workspace for portfolio-level analysis, single-customer scoring, batch CSV scoring, segment review and cost-sensitive threshold exploration.

## What is included

- **Command centre:** customer and churn KPIs, country churn comparison, LightGBM feature importance, segment distribution and saved-model validation metrics.
- **Customer scoring:** editable customer profile, live country/city options, automatic notebook-style feature engineering and churn probability output.
- **Batch scoring:** upload raw customer CSV rows, score the portfolio, inspect the results and export a retention worklist.
- **Segments:** segment size, churn rate, lifetime value, purchasing behaviour and the labels exported by the notebook.
- **Threshold lab:** replay the original holdout split and compare the expected cost of missed churners and unnecessary retention offers.
- **Data explorer:** feature distributions, source-data missingness and raw/model-ready data previews.
- **Deployment files:** pinned Python dependencies, Linux package declaration, Streamlit theme and Dockerfile.

## Project structure

```text
churn_ai_deployable/
├── app.py
├── model_utils.py
├── requirements.txt
├── packages.txt
├── Dockerfile
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── artifacts/
│   ├── best_lightgbm.pkl
│   ├── standard_scaler.pkl
│   └── feature_columns.pkl
├── data/
│   ├── ecommerce_customer_churn_dataset.csv
│   └── model_ready_dataset.csv
├── research/
│   ├── customer_segmentation_churn_prediction.ipynb
│   └── findings.md
└── tests/
    └── test_model_utils.py
```

## Run locally

Use Python 3.11. The supplied StandardScaler pickle was created with scikit-learn 1.3.2, so the pinned versions in `requirements.txt` are intentional. The app pins Streamlit 1.66.0 and uses the current `width="stretch"` layout API.

### macOS / Linux

```bash
cd churn_ai_deployable
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit (normally `http://localhost:8501`).

### Windows PowerShell

```powershell
cd churn_ai_deployable
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the **contents** of this project folder.
2. Confirm that `app.py`, `artifacts/` and `data/` are at the repository root level as shown above.
3. In Streamlit Community Cloud, create a new app, select the repository and branch, and set the main file path to `app.py`.
4. Open **Advanced settings** and select **Python 3.11**. This is important because the supplied scaler was pickled with scikit-learn 1.3.2.
5. Deploy. Community Cloud installs Python packages from `requirements.txt` and Linux packages from `packages.txt`; `.streamlit/config.toml` supplies the app theme.

Keep the three pickle files and both CSV files in the repository. Do not move them outside the project unless you also update the paths in `model_utils.py`.

## Run with Docker

```bash
docker build -t churniq .
docker run --rm -p 8501:8501 churniq
```

Then open `http://localhost:8501`.

## Findings preserved from the supplied notebook

The source cohort contains **50,000 customers**: 35,550 retained and 14,450 churned, for an observed churn rate of **28.90%**.

The saved LightGBM model was selected in the notebook after model comparisons. Its recorded 10,000-row holdout results at a 0.50 probability threshold are:

| Metric | Result |
|---|---:|
| Accuracy | 92.11% |
| Precision | 91.24% |
| Recall | 80.42% |
| F1 score | 85.49% |
| ROC-AUC | 92.96% |

The app reconstructs the notebook's stratified `train_test_split(test_size=0.2, random_state=42)` from the exported model-ready data, restores the saved `Anomaly_Flag` column to the training-time feature name and checks these metrics against the model artifact. The reconstructed values match the notebook output.

The notebook's scenario calculation estimates approximately **$2.04M** in retained lifetime value under an assumed 10% relative churn reduction. This is a scenario calculation, not a measured or guaranteed financial outcome.

## Important model and data caveats

These caveats are included in the application because they affect how its predictions should be interpreted.

1. **Anomaly-feature target leakage:** in the notebook, the IsolationForest detector is fitted before the train/test split using the original numeric columns, which include `Churned`. This can leak target information into the anomaly feature and make holdout metrics optimistic. The detector itself was not exported as a fitted artifact. The app therefore restores the historical `Anomaly_Flag` for holdout evaluation, but uses a neutral value of `0` for a newly entered customer or a batch row without an anomaly flag. Retrain this feature without `Churned`, then retrain and re-export the model before using scores for consequential production decisions.
2. **Feature artifact mismatch:** `feature_columns.pkl` contains 93 post-analysis columns, including segmentation outputs, while the scaler and LightGBM model require 91 features. The app correctly takes feature names and order from `standard_scaler.pkl` and rebuilds the engineered values before transforming a customer profile.
3. **Segment label imbalance:** the exported model-ready file contains 49,842 `Cost-Sensitive` rows and 158 `At-Risk Bargain Hunters` rows. The segment mapping/assignment needs review before the labels are used for segment-specific campaigns. The UI displays this warning rather than presenting the segments as balanced.
4. **SMOTE and probability calibration:** the saved LightGBM model is trained on a SMOTE-balanced training set. Its `predict_proba` outputs may not be calibrated to the natural 28.90% cohort churn prevalence. Check calibration (for example, reliability curves and Brier score) on an untouched, naturally distributed validation set before treating the outputs as real-world probabilities.
5. **Threshold costs are assumptions:** the default $500 cost for a missed churner and $50 cost for an unnecessary retention offer are the notebook's scenario values. Replace them with business-approved values for operational planning.
6. **Prediction is not causation:** feature importance is global model importance, not a causal effect or local explanation for an individual customer. The customer screen's profile signals are simple descriptive checks, not SHAP explanations.

## Batch CSV format

The batch scorer expects the original raw customer columns. It can fill absent numeric values with training-cohort medians and absent categorical values with the raw dataset's modes. It preserves any `Customer_ID` column in the exported output. A one-row CSV template can be downloaded directly from the Batch scoring tab.

For historical model-ready records, use the saved model-ready dataset and its `Anomaly_Flag` field for evaluation; do not upload that encoded dataset as a raw customer file to the batch form.

## Tests

The tests verify that the saved model and scaler load, a reconstructed complete customer row matches its exported 91-feature model row, the holdout metrics reproduce the notebook, and batch probabilities remain in the `[0, 1]` range.

```bash
python -m unittest discover -s tests -v
```

## Security note

Only use trusted model pickle files. Python pickle/joblib artifacts can execute code while loading; do not replace the artifacts with files from untrusted sources.
