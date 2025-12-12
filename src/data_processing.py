"""
Docstring for src.data_processing
"""
from typing import List, Optional
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
        """Returns a table of missing values and their percentages."""
        missing = self.df.isnull().sum()
        percent = 100 * missing / len(self.df)
        table = pd.concat([missing, percent], axis=1).rename(
            columns={0: 'Missing', 1: '%'})
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
