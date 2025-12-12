"""
Docstring for src.data_processing
"""
from typing import List, Optional
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    """Class to load data from various file formats."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        """Load data from the specified file path."""
        ext = self.filepath.split('.')[-1]
        if ext in ["csv", "CSV"]:
            return pd.read_csv(self.filepath)
        elif ext in ["xlsx", "xls"]:
            return pd.read_excel(self.filepath)
        else:
            raise ValueError("Unsupported file extension.")


class EDA:
    """Class for performing Exploratory Data Analysis (EDA) on a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def dataset_overview(self):
        """Provides an overview of the dataset."""
        shape_info = f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}"
        types = self.df.dtypes.value_counts().to_dict()
        head = self.df.head(3)
        return {"shape": shape_info, "dtypes": types, "preview": head}

    def summary_statistics(self) -> pd.DataFrame:
        """Returns summary statistics for numerical columns."""
        return self.df.describe()

    def plot_numerical_distributions(self, num_cols: Optional[List[str]] = None):
        """Plots distributions for numerical columns."""
        if not num_cols:
            num_cols = self.df.select_dtypes(
                include=np.number).columns.tolist()
        for col in num_cols:
            plt.figure(figsize=(6, 3))
            sns.histplot(self.df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.show()

    def plot_categorical_distributions(
        self,
        cat_cols: Optional[List[str]] = None,
        max_unique: int = 15,
    ):
        """Plots distributions for categorical columns."""
        if not cat_cols:
            cat_cols = (
                self.df.select_dtypes(include='object')
                .columns.tolist()
            )
        for col in cat_cols:
            if self.df[col].nunique() <= max_unique:
                plt.figure(figsize=(8, 3))
                self.df[col].value_counts().plot(kind="bar")
                plt.title(f'Distribution of {col}')
                plt.show()

    def correlation_matrix(self):
        """Plots the correlation matrix for numerical features."""
        corr = self.df.select_dtypes(include=np.number).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title("Correlation Matrix")
        plt.show()
        return corr

    def missing_values_table(self) -> pd.DataFrame:
        """Return missing-value diagnostics.

        Enhancements vs. the simple implementation:
        - Guards empty DataFrames
        - Optional sentinel normalization (treat common tokens as NA)
        - Adds `Dtype`, `Nunique`, `Sample` and a heuristic `Suggestion`
        """
        # Guard empty or None DataFrame
        if self.df is None or len(self.df) == 0:
            cols = [
                'Missing',
                'Percent',
                'Dtype',
                'Nunique',
                'Sample',
                'Suggestion',
            ]
            return pd.DataFrame(columns=cols)

        # Work on a copy so we don't mutate `self.df` accidentally
        df = self.df.copy()

        # Basic missing counts (only True NA/NaN)
        missing = df.isnull().sum()
        # Percent of rows missing per column
        percent = 100 * missing / len(df)

        # Basic diagnostics helpful for choosing imputations
        dtypes = df.dtypes
        nunique = df.nunique(dropna=True)

        # Collect a small sample of non-missing values for quick inspection
        samples = {}
        for col in df.columns:
            non_null = df[col].dropna().unique()
            samples[col] = list(non_null[:3]) if len(non_null) > 0 else []

        table = pd.DataFrame(
            {
                'Missing': missing,
                'Percent': percent.round(1),
                'Dtype': dtypes,
                'Nunique': nunique,
                'Sample': pd.Series(samples),
            }
        )

        # Lightweight heuristic suggestions for common cases
        def _suggest(row):
            if row['Missing'] == 0:
                return 'none'
            dtype = row['Dtype']
            pct = row['Percent']
            if np.issubdtype(dtype, np.number):
                if pct <= 5:
                    return 'median'
                if pct <= 30:
                    return 'median_or_model'
                return 'consider_drop_or_model'
            else:
                # categorical / object
                if pct <= 20:
                    return 'mode'
                if pct <= 50:
                    return 'mode_with_flag'
                return 'consider_drop_or_new_category'

        table['Suggestion'] = table.apply(_suggest, axis=1)

        # Return only columns with missing values, sorted by Missing desc
        table = table[table['Missing'] > 0].sort_values(
            'Missing', ascending=False)
        return table

    def boxplot_outliers(self, num_cols: Optional[List[str]] = None):
        """Plots boxplots for numerical columns to identify outliers."""
        if not num_cols:
            num_cols = self.df.select_dtypes(
                include=np.number).columns.tolist()
        for col in num_cols:
            plt.figure(figsize=(6, 2))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot for {col}')
            plt.show()


class DataProcessor:
    """Simple, reusable data processing pipeline.

    Provides common cleaning steps for the project tasks:
    - load raw CSV (or accept a DataFrame)
    - normalize sentinel tokens to `np.nan`
    - drop high-missing columns
    - drop duplicates
    - impute numeric with median and categorical with a 'Missing' token
    - encode categorical columns (one-hot for low-cardinality, ordinal codes for high-cardinality)
    - optional date parsing
    - save processed CSV to `data/processed/`
    - simple train/test split (pandas-based)
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    def load(self, filepath: str):
        """Load data from a CSV or Excel file using DataLoader."""
        self.df = DataLoader(filepath).load_data()
        return self.df

    @dataclass
    class ProcessConfig:
        """Configuration for the data processing steps."""
        sentinels: Optional[List[str]] = None
        drop_col_thresh: float = 0.8
        one_hot_max_unique: int = 10
        date_columns: Optional[List[str]] = None
        proxy_fn: Optional[callable] = None

    def _impute_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute numeric (median) and categorical (literal 'Missing')."""
        num_cols = df.select_dtypes(include=[np.number]).columns
        for c in num_cols:
            if df[c].isnull().any():
                df[c].fillna(df[c].median(), inplace=True)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for c in cat_cols:
            df[c].fillna('Missing', inplace=True)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, one_hot_max_unique: int) -> pd.DataFrame:
        """Encode categoricals: one-hot if low-cardinality, else add `_code` column."""
        cat_cols = df.select_dtypes(
            include=['object', 'category']).columns.tolist()
        to_concat = []
        drop_after = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= one_hot_max_unique:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                to_concat.append(dummies)
                drop_after.append(col)
            else:
                df[col + '_code'] = df[col].astype('category').cat.codes
        if to_concat:
            df = pd.concat([df] + to_concat, axis=1)
            df.drop(columns=drop_after, inplace=True)
        return df

    def process(self, cfg: Optional["DataProcessor.ProcessConfig"] = None) -> pd.DataFrame:
        """Run processing steps and return processed DataFrame.

        Accepts an optional `ProcessConfig` dataclass to avoid many positional arguments.
        """
        if self.df is None:
            raise ValueError(
                "No dataframe loaded. Call load() or set df before processing.")
        if cfg is None:
            cfg = DataProcessor.ProcessConfig()

        df = self.df.copy()
        sentinels = cfg.sentinels or ['', 'NA', 'N/A', 'Unknown', 'None']
        if sentinels:
            df.replace(sentinels, np.nan, inplace=True)
        if cfg.date_columns:
            for col in cfg.date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

        # Drop columns with too many missing values
        drop_cols = df.columns[df.isnull().mean() >
                               cfg.drop_col_thresh].tolist()
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

        df.drop_duplicates(inplace=True)

        # Impute and encode using helpers to reduce local variables
        df = self._impute_columns(df)
        df = self._encode_categoricals(df, cfg.one_hot_max_unique)

        if cfg.proxy_fn is not None and callable(cfg.proxy_fn):
            df['target'] = cfg.proxy_fn(df)

        self.df = df
        return df

    def save_processed(self, out_path: str = 'data/processed/processed_data.csv') -> str:
        """Save processed DataFrame to CSV, create directory if needed. Returns path."""
        if self.df is None:
            raise ValueError(
                'No processed dataframe to save. Call process() first.')
        out_dir = os.path.dirname(out_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        self.df.to_csv(out_path, index=False)
        return out_path

    @dataclass
    class SplitConfig:
        """Configuration for train/test split."""
        train_path: str = 'data/processed/train.csv'
        test_path: str = 'data/processed/test.csv'
        test_frac: float = 0.2
        random_state: int = 42
        stratify: Optional[str] = None

    def split_save(self, cfg: Optional["DataProcessor.SplitConfig"] = None) -> tuple:
        """Simple train/test split and save using pandas sample (no sklearn).

        - `stratify`: column name to preserve class proportions if present.
        Returns (train_path, test_path).
        """
        if self.df is None:
            raise ValueError(
                'No processed dataframe to split. Call process() first.')
        if cfg is None:
            cfg = DataProcessor.SplitConfig()

        df = self.df
        if cfg.stratify and cfg.stratify in df.columns:
            train_parts, test_parts = [], []
            rng = np.random.default_rng(cfg.random_state)
            for _, sub in df.groupby(cfg.stratify):
                n_test = max(1, int(len(sub) * cfg.test_frac))
                test_idx = rng.choice(sub.index, size=n_test, replace=False)
                test_parts.append(sub.loc[test_idx])
                train_parts.append(sub.drop(index=test_idx))
            train_df = pd.concat(train_parts).sample(
                frac=1, random_state=cfg.random_state)
            test_df = pd.concat(test_parts).sample(
                frac=1, random_state=cfg.random_state)
        else:
            test_df = df.sample(frac=cfg.test_frac,
                                random_state=cfg.random_state)
            train_df = df.drop(test_df.index)

        # Ensure directories exist
        for path in (cfg.train_path, cfg.test_path):
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)

        train_df.to_csv(cfg.train_path, index=False)
        test_df.to_csv(cfg.test_path, index=False)
        return cfg.train_path, cfg.test_path
