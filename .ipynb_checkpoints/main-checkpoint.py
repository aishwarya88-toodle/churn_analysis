import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data
from src.model import train_model
from src.evaluation import evaluate_models

def main():
    print("Loading data...")
    df = load_data("churn_analysis.csv")

    print("Training models...")
    models, X_test, y_test = train_model(df)

    print("Evaluating models...")
    evaluate_models(models, X_test, y_test)

if __name__ == "__main__":
    main()
