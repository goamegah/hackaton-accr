import pandas as pd
import numpy as np


class PreProcessing:
    def __init__(self):
        pass

    def dropna(self, df: pd.DataFrame, axis=0, how='any', subset=None) -> pd.DataFrame:
        """
        Drop rows or columns with missing values.
        :param df: DataFrame
        :param axis: 0 to drop rows, 1 to drop columns
        :param how: 'any' to drop if any NaN, 'all' to drop if all values are NaN
        :param subset: Specify columns/rows to consider
        :return: DataFrame without NaN
        """
        return df.dropna(axis=axis, how=how, subset=subset)

    def imputation(self, df: pd.DataFrame, strategy="mean", fill_value=None) -> pd.DataFrame:
        """
        Impute missing values in the DataFrame.
        :param df: DataFrame with missing values
        :param strategy: Strategy for imputation ('mean', 'median', 'mode', or 'constant')
        :param fill_value: Value to use for 'constant' strategy
        :return: DataFrame with imputed values
        """
        df = df.copy()

        qualitatives = df.select_dtypes(include=[object]).columns
        quantitatives = df.select_dtypes(include=[np.number]).columns

        for col in qualitatives:
            mode_value = df[col].mode()[0] if not df[col].mode().empty else None
            df.loc[:, col] = df[col].fillna(mode_value)
        for col in quantitatives:
            if strategy == "mean":
                df.loc[:, col] = df[col].fillna(df[col].mean())
            elif strategy == "median":
                df.loc[:, col] = df[col].fillna(df[col].median())
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("For 'constant' strategy, a fill_value must be provided.")
                df.loc[:, col] = df[col].fillna(fill_value)

        return df

    def knn_imputation(self, df: pd.DataFrame, k=5) -> pd.DataFrame:
        """
        Impute missing values using K-Nearest Neighbors (KNN).
        :param df: DataFrame
        :param k: Number of neighbors to consider
        :return: DataFrame with imputed values
        """
        from scipy.spatial import distance

        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            missing_indices = df[df[col].isnull()].index

            for idx in missing_indices:
                # Calculate distances from the row with NaN to others
                row_values = numeric_df.loc[idx].dropna()
                distances = numeric_df.drop(index=idx).apply(
                    lambda x: distance.euclidean(x.dropna(), row_values) if not x.isnull().any() else np.inf,
                    axis=1
                )
                # Find k nearest neighbors and impute mean value
                nearest_neighbors = distances.nsmallest(k).index
                imputed_value = numeric_df.loc[nearest_neighbors, col].mean()
                df.at[idx, col] = imputed_value
        return df

    def forward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward fill missing values.
        :param df: DataFrame
        :return: DataFrame with forward-filled values
        """
        return df.fillna(method='ffill')

    def backward_fill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Backward fill missing values.
        :param df: DataFrame
        :return: DataFrame with backward-filled values
        """
        return df.fillna(method='bfill')

    def random_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly impute missing values using existing values in the column.
        :param df: DataFrame
        :return: DataFrame with randomly imputed values
        """
        for col in df.columns:
            missing_indices = df[df[col].isnull()].index
            if not missing_indices.empty:
                random_values = df[col].dropna().sample(len(missing_indices), replace=True).values
                df.loc[missing_indices, col] = random_values
        return df
