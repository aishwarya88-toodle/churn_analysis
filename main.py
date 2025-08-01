import os
import sys

# Ensure the 'src' folder is discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from model import train_model
from evaluation import evaluate_models

def main():
    print("Loading data...")
    file_path = os.path.join("data", "churn_analysis.csv")
    df = load_data(file_path)

    print("Training models...")
    models, X_test, y_test = train_model(df)

    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
