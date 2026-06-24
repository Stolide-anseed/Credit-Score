"""Runtime preprocessing for Credit Score model inference.

Use this module in production/inference code:

    from scr.preprocessing import load_preprocessor, transform_for_model

    preprocessor = load_preprocessor("model/preprocessor.joblib")
    X_model = transform_for_model(raw_rows, preprocessor)
    predictions = trained_model.predict(X_model)

The fitted preprocessor must be saved from the training data. It stores clip bounds,
customer profiles, loan type columns, imputers, scaler and one-hot encoder categories.
"""

from pathlib import Path
import json
import joblib
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Credit_Score"
SCORE_ORDER = ["Poor", "Standard", "Good"]
SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}
GROUP_COLUMN = "Customer_ID"
TEST_SIZE = 0.2
SPLIT_STRATEGY = "customer_id_group_shuffle"

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data" / "train.csv").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

MISSING_MARKERS = {
    "",
    "_",
    "__",
    "___",
    "_______",
    "!@9#%8",
    "nan",
    "NaN",
    "None",
    "<NA>",
}

NUMERIC_LIKE_COLS = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

EXPECTED_RANGES = {
    "Age": (18, 100),
    "Annual_Income": (0, 1_000_000),
    "Monthly_Inhand_Salary": (0, 100_000),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Delay_from_due_date": (0, 90),
    "Num_of_Delayed_Payment": (0, 100),
    "Num_Credit_Inquiries": (0, 100),
    "Outstanding_Debt": (0, 20_000),
    "Total_EMI_per_month": (0, 20_000),
    "Amount_invested_monthly": (0, 20_000),
    "Monthly_Balance": (0, 10_000),
    "Credit_History_Age_Months": (0, 600),
}

DROP_AFTER_CLEANING = [
    "ID",
    "Customer_ID",
    "Name",
    "SSN",
    "Credit_History_Age",
    "Type_of_Loan",
    TARGET,
]

PROFILE_NUMERIC_COLS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

PROFILE_CATEGORICAL_COLS = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

MONTH_ORDER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

ENGINEERED_FEATURES = [
    "debt_to_annual_income",
    "debt_to_monthly_salary",
    "emi_to_monthly_salary",
    "investment_to_monthly_salary",
    "balance_to_monthly_salary",
    "available_income_after_emi",
    "available_income_after_emi_and_investment",
    "delayed_payment_ratio",
    "avg_delay_per_delayed_payment",
    "inquiries_per_credit_account",
    "credit_cards_per_bank_account",
    "loans_per_bank_account",
    "credit_history_years",
    "credit_age_per_loan",
    "total_credit_products",
    "loan_diversity_ratio",
    "has_negative_payment_history",
    "high_utilization_flag",
    "low_balance_flag",
    "debt_per_credit_product",
    "interest_debt_pressure",
    "credit_inquiries_per_history_year",
    "credit_pressure_utilized_debt",
    "free_cash_flow_proxy",
    "debt_service_and_investment_to_salary",
    "secured_loan_count",
    "unsecured_loan_count",
    "has_no_reported_loan_type",
    "credit_mix_ordinal",
    "payment_of_min_amount_ordinal",
    "payment_min_amount_yes_flag",
    "is_missing_Type_of_Loan",
    "is_missing_Credit_History_Age",
    "is_missing_Monthly_Inhand_Salary",
    "is_missing_Credit_Mix",
    "missing_marker_count",
    "is_anomaly_Age",
    "is_anomaly_Delay_from_due_date",
    "anomaly_count",
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "Payment_Spend_Level",
    "Payment_Value_Size",
]

SECURED_LOAN_TYPES = {
    "Auto Loan",
    "Home Equity Loan",
    "Mortgage Loan",
}

UNSECURED_LOAN_TYPES = {
    "Credit-Builder Loan",
    "Debt Consolidation Loan",
    "Payday Loan",
    "Personal Loan",
    "Student Loan",
}

CREDIT_MIX_ORDINAL = {
    "Bad": 0.0,
    "Standard": 1.0,
    "Good": 2.0,
}

PAYMENT_MIN_AMOUNT_ORDINAL = {
    "Yes": 0.0,
    "NM": 1.0,
    "No": 2.0,
}

MISSING_FLAG_SOURCE_COLS = [
    "Type_of_Loan",
    "Credit_History_Age",
    "Monthly_Inhand_Salary",
    "Credit_Mix",
]

ANOMALY_FLAG_COLS = [
    "Age",
    "Delay_from_due_date",
]


def normalize_missing_markers(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(MISSING_MARKERS))
    return cleaned


def parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = normalize_missing_markers(series)
    cleaned = cleaned.str.replace("_", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def parse_credit_history_to_months(series: pd.Series) -> pd.Series:
    cleaned = normalize_missing_markers(series)
    parts = cleaned.str.extract(
        r"(?:(\d+)\s+Years?)?\s*(?:and\s*)?(?:(\d+)\s+Months?)?",
        expand=True,
    )
    years = pd.to_numeric(parts[0], errors="coerce")
    months = pd.to_numeric(parts[1], errors="coerce")
    parsed = years.fillna(0).mul(12).add(months.fillna(0))
    return parsed.where(parts.notna().any(axis=1))


def parse_loan_types(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text in MISSING_MARKERS:
        return []

    text = re.sub(r"\band\b", ",", text, flags=re.IGNORECASE)
    loan_types = []
    for part in text.split(","):
        cleaned = part.strip().strip(".")
        if cleaned and cleaned not in MISSING_MARKERS:
            loan_types.append(cleaned)
    return loan_types


def _series_mode(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return np.nan
    return values.mode(dropna=True).iloc[0]

class CreditScoreCleaner(BaseEstimator, TransformerMixin):
    # Очищает raw-строки Credit Score перед общим sklearn preprocessing.

    def __init__(
        self,
        use_customer_profiles: bool = True,
        clip_quantiles: tuple[float, float] = (0.01, 0.99),
    ):
        self.use_customer_profiles = use_customer_profiles
        self.clip_quantiles = clip_quantiles

    def fit(self, X: pd.DataFrame, y: object = None):
        data = self._basic_clean(X)
        self.feature_columns_in_ = list(X.columns)

        loan_counts = {}
        if "Type_of_Loan" in data.columns:
            for loans in data["Type_of_Loan"].apply(parse_loan_types):
                for loan in loans:
                    loan_counts[loan] = loan_counts.get(loan, 0) + 1
        self.loan_types_ = sorted(loan_counts)

        self.numeric_columns_ = [
            column
            for column in data.select_dtypes(include=[np.number]).columns
            if column not in [TARGET]
        ]

        self.clip_bounds_ = {}
        if self.clip_quantiles is not None:
            lower_q, upper_q = self.clip_quantiles
            for column in self.numeric_columns_:
                series = pd.to_numeric(data[column], errors="coerce")
                if series.notna().any():
                    lower, upper = series.quantile([lower_q, upper_q])
                    self.clip_bounds_[column] = (float(lower), float(upper))

        if self.use_customer_profiles and "Customer_ID" in data.columns:
            self.customer_numeric_profiles_ = self._fit_customer_numeric_profiles(data)
            self.customer_categorical_profiles_ = self._fit_customer_categorical_profiles(data)
        else:
            self.customer_numeric_profiles_ = pd.DataFrame()
            self.customer_categorical_profiles_ = pd.DataFrame()

        cleaned = self._apply_customer_profiles(data)
        cleaned = self._add_loan_features(cleaned)
        cleaned = self._clip_numeric_features(cleaned)
        cleaned = self._add_credit_engineering_features(cleaned)
        cleaned = self._post_clean(cleaned)
        self.output_columns_ = list(cleaned.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = self._basic_clean(X)
        data = self._apply_customer_profiles(data)
        data = self._add_loan_features(data)
        data = self._clip_numeric_features(data)
        data = self._add_credit_engineering_features(data)
        data = self._post_clean(data)

        for column in self.output_columns_:
            if column not in data.columns:
                data[column] = np.nan
        return data[self.output_columns_]

    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()

        for column in data.select_dtypes(include=["object", "string"]).columns:
            data[column] = normalize_missing_markers(data[column])

        missing_flag_columns = [column for column in MISSING_FLAG_SOURCE_COLS if column in data.columns]
        if missing_flag_columns:
            data["missing_marker_count"] = data[missing_flag_columns].isna().sum(axis=1).astype("float64")
        else:
            data["missing_marker_count"] = np.nan

        for column in MISSING_FLAG_SOURCE_COLS:
            flag_name = f"is_missing_{column}"
            if column in data.columns:
                data[flag_name] = data[column].isna().astype("float64")
            else:
                data[flag_name] = np.nan

        for column in NUMERIC_LIKE_COLS:
            if column in data.columns:
                data[column] = parse_numeric(data[column])

        if "Credit_History_Age" in data.columns:
            data["Credit_History_Age_Months"] = parse_credit_history_to_months(
                data["Credit_History_Age"]
            )

        if "Month" in data.columns:
            data["Month_Num"] = data["Month"].map(MONTH_ORDER).astype("float64")

        anomaly_count = pd.Series(0.0, index=data.index, dtype="float64")
        for column, (lower, upper) in EXPECTED_RANGES.items():
            if column in data.columns:
                series = pd.to_numeric(data[column], errors="coerce")
                anomaly_mask = series.notna() & ((series < lower) | (series > upper))
                anomaly_count = anomaly_count.add(anomaly_mask.astype("float64"), fill_value=0)
                if column in ANOMALY_FLAG_COLS:
                    data[f"is_anomaly_{column}"] = anomaly_mask.astype("float64")
                data[column] = series.mask(anomaly_mask).astype("float64")

        for column in ANOMALY_FLAG_COLS:
            flag_name = f"is_anomaly_{column}"
            if flag_name not in data.columns:
                data[flag_name] = np.nan

        data["anomaly_count"] = anomaly_count.astype("float64")
        return data

    def _fit_customer_numeric_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = [
            column
            for column in PROFILE_NUMERIC_COLS
            if column in data.columns and data[column].notna().any()
        ]
        if not columns:
            return pd.DataFrame()
        return data.groupby("Customer_ID", dropna=True)[columns].median()

    def _fit_customer_categorical_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = [
            column
            for column in PROFILE_CATEGORICAL_COLS
            if column in data.columns and data[column].notna().any()
        ]
        if not columns:
            return pd.DataFrame()
        return data.groupby("Customer_ID", dropna=True)[columns].agg(_series_mode)

    def _apply_customer_profiles(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Customer_ID" not in data.columns:
            return data

        if not self.customer_numeric_profiles_.empty:
            for column in self.customer_numeric_profiles_.columns:
                if column in data.columns:
                    profile_values = data["Customer_ID"].map(self.customer_numeric_profiles_[column])
                    data[column] = data[column].fillna(profile_values)

        if not self.customer_categorical_profiles_.empty:
            for column in self.customer_categorical_profiles_.columns:
                if column in data.columns:
                    profile_values = data["Customer_ID"].map(self.customer_categorical_profiles_[column])
                    data[column] = data[column].fillna(profile_values)

        return data

    def _add_loan_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if "Type_of_Loan" not in data.columns:
            data["Loan_Type_Count"] = np.nan
            data["secured_loan_count"] = np.nan
            data["unsecured_loan_count"] = np.nan
            data["has_no_reported_loan_type"] = np.nan
            for loan_type in self.loan_types_:
                data[f"Loan_Type__{loan_type}"] = 0.0
            return data

        parsed = data["Type_of_Loan"].apply(parse_loan_types)
        loan_sets = parsed.map(set)

        data["Loan_Type_Count"] = parsed.map(len).astype("float64")
        data["secured_loan_count"] = parsed.map(
            lambda loans: sum(loan in SECURED_LOAN_TYPES for loan in loans)
        ).astype("float64")
        data["unsecured_loan_count"] = parsed.map(
            lambda loans: sum(loan in UNSECURED_LOAN_TYPES for loan in loans)
        ).astype("float64")
        data["has_no_reported_loan_type"] = parsed.map(lambda loans: len(loans) == 0).astype("float64")

        for loan_type in self.loan_types_:
            data[f"Loan_Type__{loan_type}"] = loan_sets.map(lambda loans: float(loan_type in loans))
        return data

    def _numeric_feature(self, data: pd.DataFrame, column: str) -> pd.Series:
        if column not in data.columns:
            return pd.Series(np.nan, index=data.index, dtype="float64")
        return pd.to_numeric(data[column], errors="coerce").astype("float64")

    @staticmethod
    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.where(denominator.ne(0))
        result = numerator.divide(denominator)
        return result.replace([np.inf, -np.inf], np.nan).astype("float64")

    @staticmethod
    def _sum_known(*series_list: pd.Series) -> pd.Series:
        values = pd.concat(series_list, axis=1)
        result = values.fillna(0).sum(axis=1)
        return result.where(values.notna().any(axis=1)).astype("float64")

    def _add_credit_engineering_features(self, data: pd.DataFrame) -> pd.DataFrame:
        annual_income = self._numeric_feature(data, "Annual_Income")
        monthly_salary = self._numeric_feature(data, "Monthly_Inhand_Salary")
        bank_accounts = self._numeric_feature(data, "Num_Bank_Accounts")
        credit_cards = self._numeric_feature(data, "Num_Credit_Card")
        loans = self._numeric_feature(data, "Num_of_Loan")
        delayed_payments = self._numeric_feature(data, "Num_of_Delayed_Payment")
        credit_inquiries = self._numeric_feature(data, "Num_Credit_Inquiries")
        outstanding_debt = self._numeric_feature(data, "Outstanding_Debt")
        credit_history_months = self._numeric_feature(data, "Credit_History_Age_Months")
        total_emi = self._numeric_feature(data, "Total_EMI_per_month")
        monthly_investment = self._numeric_feature(data, "Amount_invested_monthly")
        monthly_balance = self._numeric_feature(data, "Monthly_Balance")
        delay_from_due_date = self._numeric_feature(data, "Delay_from_due_date")
        utilization = self._numeric_feature(data, "Credit_Utilization_Ratio")
        interest_rate = self._numeric_feature(data, "Interest_Rate")
        loan_type_count = self._numeric_feature(data, "Loan_Type_Count")

        total_credit_products = self._sum_known(credit_cards, loans)
        credit_history_years = credit_history_months.div(12).astype("float64")
        free_cash_flow = monthly_salary.sub(total_emi).sub(monthly_investment).astype("float64")

        data["debt_to_annual_income"] = self._safe_divide(outstanding_debt, annual_income)
        data["debt_to_monthly_salary"] = self._safe_divide(outstanding_debt, monthly_salary)
        data["emi_to_monthly_salary"] = self._safe_divide(total_emi, monthly_salary)
        data["investment_to_monthly_salary"] = self._safe_divide(monthly_investment, monthly_salary)
        data["balance_to_monthly_salary"] = self._safe_divide(monthly_balance, monthly_salary)
        data["available_income_after_emi"] = monthly_salary.sub(total_emi).astype("float64")
        data["available_income_after_emi_and_investment"] = free_cash_flow
        data["free_cash_flow_proxy"] = free_cash_flow
        data["debt_service_and_investment_to_salary"] = self._safe_divide(
            total_emi.add(monthly_investment),
            monthly_salary,
        )
        data["interest_debt_pressure"] = outstanding_debt.mul(interest_rate).div(100).astype("float64")
        data["credit_pressure_utilized_debt"] = outstanding_debt.mul(utilization).div(100).astype("float64")
        data["credit_inquiries_per_history_year"] = self._safe_divide(
            credit_inquiries,
            credit_history_years.add(1),
        )
        data["delayed_payment_ratio"] = self._safe_divide(
            delayed_payments,
            credit_history_months,
        )
        data["avg_delay_per_delayed_payment"] = self._safe_divide(
            delay_from_due_date,
            delayed_payments,
        )
        data["inquiries_per_credit_account"] = self._safe_divide(
            credit_inquiries,
            total_credit_products,
        )
        data["credit_cards_per_bank_account"] = self._safe_divide(
            credit_cards,
            bank_accounts,
        )
        data["loans_per_bank_account"] = self._safe_divide(loans, bank_accounts)
        data["credit_history_years"] = credit_history_years
        data["credit_age_per_loan"] = self._safe_divide(credit_history_months, loans)
        data["total_credit_products"] = total_credit_products
        data["loan_diversity_ratio"] = self._safe_divide(loan_type_count, loans)

        if "Credit_Mix" in data.columns:
            data["credit_mix_ordinal"] = data["Credit_Mix"].map(CREDIT_MIX_ORDINAL).astype("float64")
        else:
            data["credit_mix_ordinal"] = np.nan

        if "Payment_of_Min_Amount" in data.columns:
            payment_min = data["Payment_of_Min_Amount"]
            data["payment_of_min_amount_ordinal"] = payment_min.map(PAYMENT_MIN_AMOUNT_ORDINAL).astype("float64")
            data["payment_min_amount_yes_flag"] = payment_min.eq("Yes").astype("float64").where(payment_min.notna())
        else:
            data["payment_of_min_amount_ordinal"] = np.nan
            data["payment_min_amount_yes_flag"] = np.nan

        if "Payment_Behaviour" in data.columns:
            behaviour = data["Payment_Behaviour"].astype("string")
            data["Payment_Spend_Level"] = behaviour.str.extract(
                r"^(High|Low)_spent",
                expand=False,
            ).astype("object")
            data["Payment_Value_Size"] = behaviour.str.extract(
                r"_(Small|Medium|Large)_value_payments$",
                expand=False,
            ).astype("object")
        else:
            data["Payment_Spend_Level"] = np.nan
            data["Payment_Value_Size"] = np.nan

        has_payment_data = delayed_payments.notna() | delay_from_due_date.notna()
        has_negative_history = delayed_payments.gt(0) | delay_from_due_date.gt(0)
        data["has_negative_payment_history"] = (
            has_negative_history.astype("float64").where(has_payment_data)
        )
        data["high_utilization_flag"] = (
            utilization.ge(35).astype("float64").where(utilization.notna())
        )
        data["low_balance_flag"] = (
            monthly_balance.le(monthly_salary.mul(0.1))
            .astype("float64")
            .where(monthly_balance.notna() & monthly_salary.notna())
        )
        data["debt_per_credit_product"] = self._safe_divide(
            outstanding_debt,
            total_credit_products,
        )

        for column in ENGINEERED_FEATURES:
            data[column] = (
                pd.to_numeric(data[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .astype("float64")
            )
        return data

    def _clip_numeric_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for column, (lower, upper) in self.clip_bounds_.items():
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce").clip(lower, upper)
        return data

    def _post_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=[column for column in DROP_AFTER_CLEANING if column in data.columns])

        for column in data.columns:
            if pd.api.types.is_string_dtype(data[column]) or data[column].dtype == object:
                data[column] = data[column].astype("object").where(data[column].notna(), np.nan)

        return data

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Пропуск")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

column_processor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
        (
            "cat",
            categorical_pipeline,
            make_column_selector(dtype_include=["object", "string", "category"]),
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
column_processor.set_output(transform="pandas")

preprocessing_pipeline = Pipeline(
    steps=[
        ("cleaner", CreditScoreCleaner(use_customer_profiles=True)),
        ("columns", column_processor),
    ]
)



DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "model" / "preprocessor.joblib"


def build_column_processor() -> ColumnTransformer:
    """Create a fresh sklearn column processor."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Пропуск")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    processor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            (
                "cat",
                categorical_pipeline,
                make_column_selector(dtype_include=["object", "string", "category"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    processor.set_output(transform="pandas")
    return processor


def build_preprocessing_pipeline(use_customer_profiles: bool = True) -> Pipeline:
    """Create an unfitted preprocessing pipeline for model training."""
    return Pipeline(
        steps=[
            ("cleaner", CreditScoreCleaner(use_customer_profiles=use_customer_profiles)),
            ("columns", build_column_processor()),
        ]
    )


def split_features_target(data: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into features and encoded target."""
    if target not in data.columns:
        raise ValueError(f"Target column not found: {target}")

    X = data.drop(columns=[target])
    y = data[target].map(SCORE_MAP)

    if y.isna().any():
        unknown_values = sorted(data.loc[y.isna(), target].dropna().unique())
        raise ValueError(f"Unknown target labels: {unknown_values}")

    return X, y.astype("int64")


def fit_preprocessor(
    data: pd.DataFrame,
    target: str = TARGET,
    use_customer_profiles: bool = True,
) -> Pipeline:
    """Fit preprocessing on raw training data and return fitted pipeline."""
    X, y = split_features_target(data, target=target)
    preprocessor = build_preprocessing_pipeline(use_customer_profiles=use_customer_profiles)
    preprocessor.fit(X, y)
    return preprocessor


def group_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    group_column: str = GROUP_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows by Customer_ID so a customer cannot appear in both train and test."""
    if group_column not in X.columns:
        raise ValueError(f"Group column not found: {group_column}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=X[group_column]))

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    train_customers = set(X_train[group_column].dropna())
    test_customers = set(X_test[group_column].dropna())
    overlap = train_customers.intersection(test_customers)
    if overlap:
        raise ValueError(f"Customer leakage between train and test: {len(overlap)}")

    return X_train, X_test, y_train, y_test


def fit_preprocessor_on_group_train(
    data: pd.DataFrame,
    target: str = TARGET,
    group_column: str = GROUP_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_customer_profiles: bool = True,
) -> tuple[Pipeline, dict[str, int]]:
    """Fit preprocessor only on group-train rows, matching the notebook split."""
    X, y = split_features_target(data, target=target)
    X_train, X_test, y_train, _ = group_train_test_split(
        X,
        y,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessing_pipeline(use_customer_profiles=use_customer_profiles)
    preprocessor.fit(X_train, y_train)

    train_customers = set(X_train[group_column].dropna())
    test_customers = set(X_test[group_column].dropna())
    profile_customers = set(preprocessor.named_steps["cleaner"].customer_numeric_profiles_.index).union(
        set(preprocessor.named_steps["cleaner"].customer_categorical_profiles_.index)
    )
    profile_test_overlap = profile_customers.intersection(test_customers)
    if profile_test_overlap:
        raise ValueError(f"Customer profile leakage into test: {len(profile_test_overlap)}")

    metadata = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_customers": int(len(train_customers)),
        "test_customers": int(len(test_customers)),
        "customer_overlap": int(len(train_customers.intersection(test_customers))),
        "customer_profile_test_overlap": int(len(profile_test_overlap)),
    }
    return preprocessor, metadata


def save_preprocessor(preprocessor: Pipeline, path: str | Path = DEFAULT_ARTIFACT_PATH) -> Path:
    """Persist fitted preprocessing pipeline."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, output_path)
    return output_path


def load_preprocessor(path: str | Path = DEFAULT_ARTIFACT_PATH) -> Pipeline:
    """Load fitted preprocessing pipeline."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Preprocessor artifact not found: {artifact_path}. "
            "Fit and save it first with fit_preprocessor/save_preprocessor."
        )
    return joblib.load(artifact_path)


def transform_for_model(data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]], preprocessor: Pipeline) -> pd.DataFrame:
    """Transform raw real-time rows into the feature matrix expected by the trained model.

    Parameters
    ----------
    data:
        Raw input row(s). Accepts a pandas DataFrame, one dict, or a list of dicts.
        The target column is ignored if present.
    preprocessor:
        Fitted preprocessing pipeline loaded by load_preprocessor.
    """
    if isinstance(data, pd.DataFrame):
        raw = data.copy()
    elif isinstance(data, dict):
        raw = pd.DataFrame([data])
    else:
        raw = pd.DataFrame(data)

    if TARGET in raw.columns:
        raw = raw.drop(columns=[TARGET])

    return preprocessor.transform(raw)


def transform_with_saved_preprocessor(
    data: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> pd.DataFrame:
    """Load saved preprocessor and transform raw row(s)."""
    preprocessor = load_preprocessor(artifact_path)
    return transform_for_model(data, preprocessor)


def fit_and_save_from_csv(
    train_csv_path: str | Path,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    target: str = TARGET,
    use_customer_profiles: bool = True,
) -> Path:
    """Fit preprocessor on a raw train CSV and save the fitted artifact."""
    data = pd.read_csv(train_csv_path, low_memory=False)
    preprocessor = fit_preprocessor(
        data,
        target=target,
        use_customer_profiles=use_customer_profiles,
    )
    return save_preprocessor(preprocessor, artifact_path)


def fit_group_train_and_save_from_csv(
    train_csv_path: str | Path,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    metadata_path: str | Path | None = None,
    target: str = TARGET,
    use_customer_profiles: bool = True,
) -> tuple[Path, dict[str, int]]:
    """Fit on group-train rows and save an artifact matching current train/test CSVs."""
    data = pd.read_csv(train_csv_path, low_memory=False)
    preprocessor, metadata = fit_preprocessor_on_group_train(
        data,
        target=target,
        use_customer_profiles=use_customer_profiles,
    )
    output_path = save_preprocessor(preprocessor, artifact_path)

    if metadata_path is not None:
        metadata_output = Path(metadata_path)
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return output_path, metadata

