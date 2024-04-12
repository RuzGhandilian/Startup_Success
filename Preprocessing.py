import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class Preprocessing:
    def __init__(self):
        # Initialize attributes
        self.to_numeric = None
        self.pure_categorical = None
        self.dropped_cols = []
        self.categorical_cols = None
        self.numerical_cols = None
        self.median_values = None
        self.mode_values = None
        self.scaler = None
        self.oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def fit(self, df):
        # Replace 'No Info' with NaN
        df.replace('No Info', np.nan, inplace=True)

        # Find and convert numerical columns
        self.numerical_cols = list(df.select_dtypes(include=['number']).columns)
        df[self.numerical_cols] = df[self.numerical_cols].apply(pd.to_numeric, errors='coerce')

        # Handle categorical NaN values by filling with mode
        self.categorical_cols = df.select_dtypes(include=['object']).columns
        self.mode_values = df[self.categorical_cols].mode().iloc[0]

        # Convert remaining object columns to numeric if possible
        self.pure_categorical = []
        self.to_numeric = []

        for col in self.categorical_cols:
            try:
                pd.to_numeric(df[col])
                self.to_numeric.append(col)
            except ValueError:
                self.pure_categorical.append(col)

        self.all_numeric = self.numerical_cols + self.to_numeric

        # Automatically remove columns with > 30 unique values
        self.dropped_cols = [col for col in self.pure_categorical if df[col].nunique() > 30]

        # Remove dropped columns from categorical columns
        self.pure_categorical = [col for col in self.pure_categorical if col not in self.dropped_cols]

        # Fit OneHotEncoder
        self.oh_encoder.fit(df[self.pure_categorical])

        # Compute median values for numerical columns
        self.median_values = df[self.numerical_cols].median()

        # Store scaler for numerical columns
        self.scaler = StandardScaler().fit(df[self.all_numeric])

    def transform(self, df):
        # Replace 'No Info' with NaN
        df.replace('No Info', np.nan, inplace=True)

        # Handle missing values in categorical columns using mode computed during fit
        for col in self.categorical_cols:
            df[col].fillna(self.mode_values[col], inplace=True)

        # Handle missing values in numerical columns using median computed during fit
        df.fillna(self.median_values, inplace=True)

        # Convert object columns to numeric if possible
        for col in self.to_numeric:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

        # Standardize numerical variables
        df[self.all_numeric] = self.scaler.transform(df[self.all_numeric])

        # Select relevant columns for transformation
        df = df[[col for col in df.columns if (col in self.pure_categorical or col in self.all_numeric)]]

        # One-hot encode categorical variables
        dummies = self.oh_encoder.transform(df[self.pure_categorical])
        df = df.drop(self.pure_categorical, axis=1)

        # Concatenate one-hot encoded features with the original DataFrame
        df_dummies = pd.DataFrame(dummies)
        df_dummies.reset_index(drop=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df_dummies], axis=1)

        # Convert column names to string
        df.columns = df.columns.astype(str)

        return df
