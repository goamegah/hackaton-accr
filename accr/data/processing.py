import numpy as np
import pandas as pd
import functools as ft
from accr.data.preprocessing import PreProcessing


class Processing:
    def __init__(self):
        pass

    def preprocessing(self, dataset: pd.DataFrame, method="drop", strategy="mean", fill_value=None, k=5, subset=None) -> pd.DataFrame:
        """
        Preprocess the input dataset by either dropping missing values or imputing them.
        """
        preprocess = PreProcessing()

        if method == "drop" and isinstance(dataset, pd.DataFrame):
            return preprocess.dropna(dataset, subset=subset)

        elif method == "imputation" and isinstance(dataset, pd.DataFrame):
            return preprocess.imputation(dataset, strategy=strategy, fill_value=fill_value)

        elif method == "knn" and isinstance(dataset, pd.DataFrame):
            return preprocess.knn_imputation(dataset, k=k)

        elif method == "forward_fill" and isinstance(dataset, pd.DataFrame):
            return preprocess.forward_fill(dataset)

        elif method == "backward_fill" and isinstance(dataset, pd.DataFrame):
            return preprocess.backward_fill(dataset)

        elif method == "random" and isinstance(dataset, pd.DataFrame):
            return preprocess.random_imputation(dataset)

        else:
            raise ValueError("Invalid method. Choose from 'drop', 'imputation', 'knn', 'forward_fill', "
                             "'backward_fill', or 'random'.")

    def summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the input DataFrame, including the number of missing values and constant features.
        """
        constant_features = [col for col in df.columns if df[col].nunique() <= 1]

        return {
            "Missing Values": df.isna().sum(),
            "Constant Features": constant_features
        }

    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all constant features from the input DataFrame.
        """
        constant_features = self.summary(df)["Constant Features"]
        return df.drop(columns=constant_features)

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        """
        Remove all outliers from the input DataFrame using the specified threshold.
        """
        qualitatives = df.select_dtypes(include=[object]).columns
        quantitatives = [c for c in df.columns if c not in qualitatives]

        def quantile(s):
            """Identify outliers using quantiles."""
            assert isinstance(s, pd.Series)
            q_025 = s.quantile(.25)
            q_075 = s.quantile(.75)
            iq = q_075 - q_025
            return ~((s > q_075 + threshold * iq) | (s < q_025 - threshold * iq))

        mask = ft.reduce(lambda x, y: x & y, [quantile(df[col]) for col in quantitatives])
        return df.loc[mask].copy()

    def transform_categorical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        """
        Transforms categorical variables in the DataFrame to ordinal or one-hot encoded format.
        """
        qualitatives = df.select_dtypes(include=[object]).columns
        ordinal_cols = []
        if not qualitatives.empty:
            if l_order:
                if not set(l_order).issubset(set(qualitatives)):
                    raise ValueError(f"{l_order} is not included in {list(qualitatives)}")
                ordinal_cols = [c for c in qualitatives if c in l_order]
                ordinal_encoder = PreProcessing()
                ordinals = ordinal_encoder.ordinal_encode(df[ordinal_cols], l_order)
                ordinal_mapping = ordinal_encoder.mapping
            else:
                ordinals, ordinal_mapping = None, None

        nominal_encoder = PreProcessing()
        nominal_cols = [c for c in qualitatives if c not in ordinal_cols]
        nominals = nominal_encoder.one_hot_encode(df[nominal_cols], drop=drop)
        nominal_mapping = nominal_encoder.mapping

        if ordinal_cols:
            column_names = list(ordinal_cols) + list(nominal_mapping)
            return {
                "numericals": pd.DataFrame(
                    data=np.hstack([ordinals, nominals]),
                    columns=column_names),
                "mapping": dict(zip(ordinal_cols, ordinal_mapping))
            }
        else:
            return {"numericals": pd.DataFrame(data=nominals, columns=list(nominal_mapping))}

    def df_to_numerical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        """
        Converts the input DataFrame to a numerical representation by transforming its categorical variables.
        """
        numericals_columns = self.transform_categorical(df, l_order, drop)
        qualitatives = df.select_dtypes(include=[object]).columns
        quantitatives = [c for c in df.columns if c not in qualitatives]

        df_transform = numericals_columns["numericals"].reset_index(drop=True) \
            .join(df[quantitatives].reset_index(drop=True), how="inner")

        result = {"df_transform": df_transform}
        if "mapping" in numericals_columns:
            result["ordinal_mapping"] = numericals_columns["mapping"]

        return result

    def replace_missing_values(self, df: pd.DataFrame, missing_values=None) -> pd.DataFrame:
        """
        Replace missing or invalid values with NaN in the DataFrame.
        """
        if missing_values is None:
            missing_values = [None, 0, -1, ".", "-1", "0"]

        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].str.strip().replace(missing_values, np.nan)
            else:
                df[col] = df[col].replace(missing_values, np.nan)

        return df
