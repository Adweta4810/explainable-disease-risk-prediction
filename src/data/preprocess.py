import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_ckd_data(filepath: str) -> pd.DataFrame:
    """Load CKD raw dataset."""
    return pd.read_csv(filepath)


def preprocess_ckd(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess CKD dataset."""
    ckd_df = df.copy()

    # Replace dirty missing values like ? with NaN
    ckd_df = ckd_df.replace(r'^\s*\?\s*$', np.nan, regex=True)

    # Remove leading/trailing spaces from text columns
    for col in ckd_df.select_dtypes(include=['object']).columns:
        ckd_df[col] = ckd_df[col].astype(str).str.strip()

    # Drop ID column if present
    if 'id' in ckd_df.columns:
        ckd_df.drop(columns=['id'], inplace=True)

    # Convert numeric-like columns
    numeric_like_cols = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
    ]
    for col in numeric_like_cols:
        if col in ckd_df.columns:
            ckd_df[col] = pd.to_numeric(ckd_df[col], errors='coerce')

    # Standardize text values
    for col in ckd_df.select_dtypes(include=['object']).columns:
        ckd_df[col] = ckd_df[col].str.lower().str.strip()

    # Explicit mappings
    mapping_dict = {
        'rbc': {'normal': 0, 'abnormal': 1},
        'pc': {'normal': 0, 'abnormal': 1},
        'pcc': {'notpresent': 0, 'present': 1},
        'ba': {'notpresent': 0, 'present': 1},
        'htn': {'no': 0, 'yes': 1},
        'dm': {'no': 0, 'yes': 1, ' yes': 1, '\tyes': 1, '\tno': 0},
        'cad': {'no': 0, 'yes': 1},
        'appet': {'poor': 0, 'good': 1},
        'pe': {'no': 0, 'yes': 1},
        'ane': {'no': 0, 'yes': 1},
        'classification': {'notckd': 0, 'ckd': 1, 'ckd\t': 1}
    }

    for col, mapping in mapping_dict.items():
        if col in ckd_df.columns:
            ckd_df[col] = ckd_df[col].replace(mapping)

    # Fill missing numeric values with median
    for col in ckd_df.select_dtypes(include=['float64', 'int64']).columns:
        ckd_df[col] = ckd_df[col].fillna(ckd_df[col].median())

    return ckd_df


def save_ckd_cleaned(df: pd.DataFrame, filepath: str) -> None:
    """Save cleaned CKD dataset."""
    df.to_csv(filepath, index=False)


def load_diabetes_data(filepath: str) -> pd.DataFrame:
    """Load diabetes raw dataset."""
    return pd.read_csv(filepath)


def preprocess_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess diabetes dataset."""
    diabetes_df = df.copy()

    cols_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace biologically invalid zeros with NaN
    for col in cols_with_invalid_zeros:
        diabetes_df[col] = diabetes_df[col].replace(0, np.nan)

    # Impute with median
    for col in cols_with_invalid_zeros:
        diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

    # Remove duplicates
    diabetes_df = diabetes_df.drop_duplicates()

    # Outlier clipping
    cols_to_clip = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_clip:
        Q1 = diabetes_df[col].quantile(0.25)
        Q3 = diabetes_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        diabetes_df[col] = diabetes_df[col].clip(lower=lower, upper=upper)

    return diabetes_df


def scale_diabetes_features(df: pd.DataFrame, target_col: str = "Outcome") -> pd.DataFrame:
    """Scale diabetes numerical features and return processed dataframe."""
    processed_df = df.copy()

    X = processed_df.drop(target_col, axis=1)
    y = processed_df[target_col]

    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    final_df = pd.concat([X, y], axis=1)
    return final_df


def save_diabetes_cleaned(df: pd.DataFrame, filepath: str) -> None:
    """Save cleaned diabetes dataset."""
    df.to_csv(filepath, index=False)