import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    
    print("Dataset loaded shape:", df.shape)
    print(df['Churn'].value_counts(dropna=False))

    # SAFETY: ensure no NaN in target
    df = df.dropna(subset=['Churn'])

    return df
