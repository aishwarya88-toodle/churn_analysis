import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['customerID', 'IsSenior'], axis=1, errors='ignore')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    return df
