from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

def preprocess_data(df, target_column='Churn'):
    X = df.drop(columns=[target_column])
    y = df[target_column].map({'Yes': 1, 'No': 0})

    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])

    return X, y, preprocessor
