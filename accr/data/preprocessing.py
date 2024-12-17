import pandas as pd
import numpy as np

class PreProcessing:
    def __init__(self):
        pass

    # Method to drop rows with missing values
    def dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna()

    def imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sélectionner les colonnes de type object, supposées être qualitatives/catégorielles
        qualitatives = df.select_dtypes(include=[object]).columns
        # Remplir les valeurs manquantes dans les colonnes qualitatives avec la mode, et dans les colonnes
        # quantitatives avec la moyenne
        return df.apply(lambda s: s.fillna(s.mode()[0]) if s.name in qualitatives else s.fillna(s.mean()), axis=0)
