"""Model/data utilities for the Customer Churn Intelligence Streamlit app.

The functions in this module deliberately keep the saved LightGBM model and
StandardScaler as the source of truth for prediction feature order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

RAW_NUMERIC_FIELDS = [
    "Age",
    "Membership_Years",
    "Login_Frequency",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Cart_Abandonment_Rate",
    "Wishlist_Items",
    "Total_Purchases",
    "Average_Order_Value",
    "Days_Since_Last_Purchase",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Email_Open_Rate",
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Social_Media_Engagement_Score",
    "Mobile_App_Usage",
    "Payment_Method_Diversity",
    "Lifetime_Value",
    "Credit_Balance",
]

# UI constraints are intentionally aligned to the ranges used by the supplied
# Streamlit app and the observed source data. The preprocessing layer clips
# the five IQR-capped fields to the ranges in the model-ready training data.
INPUT_SPECS = {
    "Age": {"label": "Age", "min": 18.0, "max": 80.0, "step": 1.0, "format": "%.0f"},
    "Membership_Years": {"label": "Membership years", "min": 0.0, "max": 10.0, "step": 0.1, "format": "%.1f"},
    "Login_Frequency": {"label": "Logins per month", "min": 0.0, "max": 40.0, "step": 1.0, "format": "%.0f"},
    "Session_Duration_Avg": {"label": "Average session duration (min)", "min": 0.0, "max": 60.0, "step": 0.5, "format": "%.1f"},
    "Pages_Per_Session": {"label": "Pages per session", "min": 1.0, "max": 25.0, "step": 0.5, "format": "%.1f"},
    "Cart_Abandonment_Rate": {"label": "Cart abandonment rate (%)", "min": 0.0, "max": 100.0, "step": 1.0, "format": "%.1f"},
    "Wishlist_Items": {"label": "Wishlist items", "min": 0.0, "max": 20.0, "step": 1.0, "format": "%.0f"},
    "Mobile_App_Usage": {"label": "Mobile app usage (hrs/month)", "min": 0.0, "max": 80.0, "step": 0.5, "format": "%.1f"},
    "Social_Media_Engagement_Score": {"label": "Social media engagement score", "min": 0.0, "max": 100.0, "step": 1.0, "format": "%.1f"},
    "Total_Purchases": {"label": "Purchases (12 months)", "min": 0.0, "max": 60.0, "step": 1.0, "format": "%.1f"},
    "Average_Order_Value": {"label": "Average order value ($)", "min": 10.0, "max": 300.0, "step": 1.0, "format": "%.2f"},
    "Days_Since_Last_Purchase": {"label": "Days since last purchase", "min": 0.0, "max": 365.0, "step": 1.0, "format": "%.0f"},
    "Discount_Usage_Rate": {"label": "Discount usage rate (%)", "min": 0.0, "max": 100.0, "step": 1.0, "format": "%.1f"},
    "Returns_Rate": {"label": "Returns rate (%)", "min": 0.0, "max": 100.0, "step": 1.0, "format": "%.1f"},
    "Lifetime_Value": {"label": "Lifetime value ($)", "min": 100.0, "max": 6000.0, "step": 10.0, "format": "%.2f"},
    "Credit_Balance": {"label": "Credit balance (loyalty points)", "min": 0.0, "max": 6000.0, "step": 10.0, "format": "%.0f"},
    "Payment_Method_Diversity": {"label": "Payment methods used", "min": 1.0, "max": 6.0, "step": 1.0, "format": "%.0f"},
    "Email_Open_Rate": {"label": "Email open rate (%)", "min": 0.0, "max": 100.0, "step": 1.0, "format": "%.1f"},
    "Customer_Service_Calls": {"label": "Customer service calls", "min": 0.0, "max": 20.0, "step": 1.0, "format": "%.0f"},
    "Product_Reviews_Written": {"label": "Product reviews written", "min": 0.0, "max": 20.0, "step": 1.0, "format": "%.0f"},
}

IQR_CAPPED_FIELDS = [
    "Lifetime_Value",
    "Credit_Balance",
    "Average_Order_Value",
    "Days_Since_Last_Purchase",
    "Returns_Rate",
]
RATIO_FEATURES = [
    "Frequency",
    "Support_Intensity",
    "Purchase_to_Login_Ratio",
    "Discount_to_Order_Value",
    "Return_to_Purchase_Ratio",
    "Lifetime_Value_Squared",
    "Cross_Channel_Engagement",
]


def load_artifacts(artifact_dir: Path | str = ARTIFACT_DIR):
    """Load the model bundle and validate feature compatibility."""
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / "best_lightgbm.pkl"
    scaler_path = artifact_dir / "standard_scaler.pkl"
    feature_path = artifact_dir / "feature_columns.pkl"
    missing = [p.name for p in (model_path, scaler_path, feature_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s): {', '.join(missing)}. "
            f"Expected them inside: {artifact_dir}"
        )

    # Pickles are version-sensitive; requirements.txt pins the original sklearn
    # version used by the supplied scaler (1.3.2).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Trying to unpickle estimator.*")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(feature_path)

    if isinstance(features, (pd.Series, np.ndarray)):
        features = list(features)
    expected = list(getattr(scaler, "feature_names_in_", features))
    model_n = getattr(model, "n_features_in_", len(expected))
    if model_n != len(expected):
        raise ValueError(
            f"Model/scaler feature mismatch: model expects {model_n}, scaler exposes {len(expected)}."
        )
    return model, scaler, list(features)


def load_datasets(data_dir: Path | str = DATA_DIR):
    """Load the raw and model-ready datasets, validating required columns."""
    data_dir = Path(data_dir)
    raw_path = data_dir / "ecommerce_customer_churn_dataset.csv"
    model_path = data_dir / "model_ready_dataset.csv"
    missing = [p.name for p in (raw_path, model_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset file(s): {', '.join(missing)}. Expected them inside: {data_dir}"
        )
    raw_df = pd.read_csv(raw_path)
    model_df = pd.read_csv(model_path)
    for col in ("Churned", "Segment_Name", "Lifetime_Value"):
        if col not in model_df.columns:
            raise ValueError(f"The model-ready dataset is missing required column: {col}")
    if "Churned" not in raw_df.columns:
        raise ValueError("The raw dataset is missing the Churned target column.")
    return raw_df, model_df


def _numeric_value(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return float(fallback)


def _profile_value(model_df: pd.DataFrame, column: str, operation: str, q: float = 0.999) -> float:
    if column not in model_df.columns:
        return 0.0
    series = pd.to_numeric(model_df[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    if operation == "median":
        return float(series.median())
    if operation == "min":
        return float(series.min())
    if operation == "max":
        return float(series.max())
    if operation == "quantile":
        return float(series.quantile(q))
    raise ValueError(f"Unsupported profile operation: {operation}")


def training_profile_stats(model_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Precompute medians, clipping bounds and engineered-feature caps once."""
    medians = {
        column: _profile_value(model_df, column, "median")
        for column in RAW_NUMERIC_FIELDS
    }
    clip_bounds = {
        column: (
            _profile_value(model_df, column, "min"),
            _profile_value(model_df, column, "max"),
        )
        for column in IQR_CAPPED_FIELDS
    }
    q999 = {
        column: _profile_value(model_df, column, "quantile", 0.999)
        for column in RATIO_FEATURES
    }
    returns_median = _profile_value(model_df, "Returns_Rate", "median")
    return {"medians": medians, "clip_bounds": clip_bounds, "q999": q999, "returns_median": returns_median}


def default_customer(
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    profile_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Create a complete, sensible example customer from training medians/modes."""
    stats = profile_stats or training_profile_stats(model_df)
    customer: dict[str, Any] = dict(stats["medians"])
    for column in ("Gender", "Country", "City", "Signup_Quarter"):
        if column in raw_df.columns:
            mode = raw_df[column].dropna().mode()
            customer[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
    return customer


def _normalise_customer(
    customer: Mapping[str, Any], raw_df: pd.DataFrame, model_df: pd.DataFrame,
    profile_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    result: dict[str, Any] = dict(customer)
    for column in RAW_NUMERIC_FIELDS:
        fallback = profile_stats["medians"].get(column, 0.0)
        result[column] = _numeric_value(result.get(column), fallback)
    for column in ("Gender", "Country", "City", "Signup_Quarter"):
        if column not in result or pd.isna(result[column]):
            if column in raw_df.columns:
                mode = raw_df[column].dropna().mode()
                result[column] = str(mode.iloc[0]) if not mode.empty else "Unknown"
            else:
                result[column] = "Unknown"
        else:
            result[column] = str(result[column])
    return result


def _clip_iqr_fields(
    values: dict[str, float], profile_stats: dict[str, dict[str, float]]
) -> None:
    """Mirror the notebook's IQR clipping using the already processed data bounds."""
    for column, (lower, upper) in profile_stats["clip_bounds"].items():
        if column in values:
            values[column] = float(np.clip(values[column], lower, upper))


def build_feature_frame(
    customer: Mapping[str, Any],
    scaler,
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    anomaly_flag: float = 0.0,
    profile_stats: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Build one model row in the exact StandardScaler feature order.

    All notebook-derived predictors are rebuilt here. A new customer receives
    anomaly_flag=0 because the notebook did not persist its IsolationForest and
    fitted it before the train/test split using a numeric matrix that included
    the Churned target. This app does not fabricate a new anomaly score.
    """
    stats = profile_stats or training_profile_stats(model_df)
    values = _normalise_customer(customer, raw_df, model_df, stats)
    numeric = {k: float(values[k]) for k in RAW_NUMERIC_FIELDS}
    _clip_iqr_fields(numeric, stats)

    def cap(value: float, col: str) -> float:
        upper = stats["q999"].get(col, 0.0)
        return float(np.clip(value, 0.0, max(upper, 0.0)))

    purchases = numeric["Total_Purchases"]
    membership_years = numeric["Membership_Years"]
    login_frequency = numeric["Login_Frequency"]
    avg_order = numeric["Average_Order_Value"]
    returns_rate = numeric["Returns_Rate"]
    discount_rate = numeric["Discount_Usage_Rate"]
    days_since_last = numeric["Days_Since_Last_Purchase"]
    session_duration = numeric["Session_Duration_Avg"]
    pages_per_session = numeric["Pages_Per_Session"]
    mobile_usage = numeric["Mobile_App_Usage"]
    social_score = numeric["Social_Media_Engagement_Score"]

    quarter_text = values["Signup_Quarter"].upper().strip()
    quarter_number = int(quarter_text[-1]) if quarter_text[-1:].isdigit() and quarter_text[-1] in "1234" else 1

    derived: dict[str, float] = dict(numeric)
    derived["Cart_Abandonment_Rate_log"] = float(np.log1p(max(numeric["Cart_Abandonment_Rate"], 0.0)))
    derived["Discount_Usage_Rate_log"] = float(np.log1p(max(discount_rate, 0.0)))
    derived["Recency"] = 365.0 - days_since_last
    derived["Frequency"] = cap(purchases / (membership_years + 1e-3), "Frequency")
    derived["Monetary"] = avg_order * purchases
    engagement_product = max(login_frequency, 0.0) * max(session_duration, 0.0) * max(pages_per_session, 0.0)
    derived["Engagement_Index"] = float(np.cbrt(engagement_product))
    derived["Support_Intensity"] = cap(
        numeric["Customer_Service_Calls"] / (purchases + 1.0), "Support_Intensity"
    )
    derived["Purchase_to_Login_Ratio"] = cap(purchases / (login_frequency + 1.0), "Purchase_to_Login_Ratio")
    derived["Discount_to_Order_Value"] = cap(discount_rate / (avg_order + 1.0), "Discount_to_Order_Value")
    derived["Return_to_Purchase_Ratio"] = cap(returns_rate / (purchases + 1.0), "Return_to_Purchase_Ratio")
    derived["Signup_Quarter_Num"] = float(quarter_number)
    derived["Lifetime_Value_Squared"] = cap(numeric["Lifetime_Value"] ** 2, "Lifetime_Value_Squared")
    return_median = stats["returns_median"]
    derived["High_Returns_Flag"] = float(returns_rate > return_median * 1.5)
    derived["Cross_Channel_Engagement"] = cap(social_score * mobile_usage, "Cross_Channel_Engagement")
    try:
        derived["anomaly_iforest"] = float(int(float(anomaly_flag) == 1.0))
    except (TypeError, ValueError):
        derived["anomaly_iforest"] = 0.0

    expected = list(getattr(scaler, "feature_names_in_", []))
    if not expected:
        raise ValueError("The saved scaler does not expose feature_names_in_; cannot guarantee feature order.")
    row = pd.Series(0.0, index=expected, dtype=float)

    for key, value in derived.items():
        if key in row.index:
            row[key] = float(value)

    # Notebook: pd.get_dummies(..., drop_first=True) for categorical columns.
    for category_field in ("Gender", "Country", "City"):
        dummy_name = f"{category_field}_{values[category_field]}"
        if dummy_name in row.index:
            row[dummy_name] = 1.0

    quarter_dummy = f"Signup_Quarter_Q{quarter_number}"
    if quarter_dummy in row.index:
        row[quarter_dummy] = 1.0

    years = membership_years
    if years <= 2:
        membership_stage = "Newbie"
    elif years <= 5:
        membership_stage = "Growing"
    elif years <= 10:
        membership_stage = "Established"
    else:
        membership_stage = "Veteran"
    membership_dummy = f"Membership_Stage_{membership_stage}"
    if membership_dummy in row.index:
        row[membership_dummy] = 1.0

    seasonal_activity = {1: "High", 2: "Moderate", 3: "Peak", 4: "Low"}[quarter_number]
    seasonal_dummy = f"Seasonal_Activity_{seasonal_activity}"
    if seasonal_dummy in row.index:
        row[seasonal_dummy] = 1.0

    return pd.DataFrame([row.to_numpy(dtype=float)], columns=expected)


def _named_model_input(model, scaled_values: np.ndarray):
    """Preserve the feature names LightGBM saw during fitting."""
    names = list(getattr(model, "feature_name_", []))
    if names and len(names) == scaled_values.shape[1]:
        return pd.DataFrame(scaled_values, columns=names)
    return scaled_values


def predict_customer(
    customer: Mapping[str, Any],
    model,
    scaler,
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    anomaly_flag: float = 0.0,
    profile_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Return churn probability, risk band and threshold decision."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")
    X = build_feature_frame(
        customer, scaler, raw_df, model_df,
        anomaly_flag=anomaly_flag, profile_stats=profile_stats,
    )
    X_scaled = _named_model_input(model, scaler.transform(X))
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_scaled)[0, 1])
    else:
        probability = float(np.asarray(model.predict(X_scaled), dtype=float)[0])
    probability = float(np.clip(probability, 0.0, 1.0))
    risk = "High" if probability >= 0.60 else "Elevated" if probability >= 0.30 else "Low"
    return {
        "probability": probability,
        "risk_level": risk,
        "flagged": probability >= float(threshold),
        "threshold": float(threshold),
        "feature_frame": X,
    }


def score_batch(
    upload_df: pd.DataFrame,
    model,
    scaler,
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Score uploaded raw customer rows with median/mode fallback for missing data."""
    if upload_df.empty:
        raise ValueError("The uploaded CSV has no customer rows.")
    scored = upload_df.copy()
    output = []
    profile_stats = training_profile_stats(model_df)
    defaults = default_customer(raw_df, model_df, profile_stats=profile_stats)
    for _, source_row in scored.iterrows():
        # Fill missing source fields from the saved training cohort once per row.
        record = dict(defaults)
        record.update(source_row.to_dict())
        anomaly = record.get("anomaly_iforest", record.get("Anomaly_Flag", 0.0))
        try:
            anomaly = float(anomaly) if pd.notna(anomaly) else 0.0
        except (TypeError, ValueError):
            anomaly = 0.0
        result = predict_customer(
            record,
            model,
            scaler,
            raw_df,
            model_df,
            threshold=threshold,
            anomaly_flag=anomaly,
            profile_stats=profile_stats,
        )
        output.append((result["probability"], result["risk_level"], result["flagged"]))
    scored["Churn_Probability"] = [x[0] for x in output]
    scored["Risk_Level"] = [x[1] for x in output]
    scored["Flagged_for_Retention"] = [x[2] for x in output]
    return scored


def holdout_evaluation(model, scaler, model_df: pd.DataFrame) -> dict[str, Any]:
    """Rebuild the notebook's original stratified 80/20 holdout and report metrics.

    The exported model-ready data stores the anomaly column as Anomaly_Flag;
    this maps it back to the training-time scaler feature name.
    """
    if "Churned" not in model_df.columns:
        raise ValueError("Churned target is required to evaluate the saved model.")
    expected = list(getattr(scaler, "feature_names_in_", []))
    if not expected:
        raise ValueError("The saved scaler has no feature names.")
    X = model_df.reindex(columns=expected, fill_value=0.0).copy()
    if "anomaly_iforest" in expected and "Anomaly_Flag" in model_df.columns:
        X["anomaly_iforest"] = pd.to_numeric(model_df["Anomaly_Flag"], errors="coerce").fillna(0).values
    X = X.reindex(columns=expected, fill_value=0.0)
    X_scaled = _named_model_input(model, scaler.transform(X))
    y = pd.to_numeric(model_df["Churned"], errors="coerce").fillna(0).astype(int).to_numpy()
    indices = np.arange(len(y))
    _, test_indices = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=42
    )
    X_test = X_scaled.iloc[test_indices] if isinstance(X_scaled, pd.DataFrame) else X_scaled[test_indices]
    y_test = y[test_indices]
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "y_true": y_test,
        "probabilities": probabilities,
        "predictions": predictions,
        "confusion": confusion_matrix(y_test, predictions),
    }


def calculate_cost_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute expected classification cost over a set of decision thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []
    for threshold in thresholds:
        predicted = probabilities >= float(threshold)
        actual = y_true == 1
        tp = int(np.sum(predicted & actual))
        fp = int(np.sum(predicted & ~actual))
        fn = int(np.sum(~predicted & actual))
        tn = int(np.sum(~predicted & ~actual))
        rows.append(
            {
                "threshold": float(threshold),
                "expected_cost": float(fn * cost_fn + fp * cost_fp),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def feature_importance_frame(model, limit: int = 12) -> pd.DataFrame:
    """Return global LightGBM split importance in descending order."""
    names = list(getattr(model, "feature_name_", []))
    values = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
    if len(names) != len(values) or len(values) == 0:
        return pd.DataFrame(columns=["Feature", "Importance"])
    frame = pd.DataFrame({"Feature": names, "Importance": values})
    return frame.sort_values("Importance", ascending=False).head(limit).reset_index(drop=True)
