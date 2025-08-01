from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
)
import matplotlib.pyplot as plt
import joblib
import os

def evaluate_models(models, X_test, y_test, output_dir='output/', model_dir='models/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    best_model_name = None
    best_auc = 0
    best_model = None

    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\nModel: {name}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC Score: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

    plt.title("ROC Curve - All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"))
    plt.close()

    joblib.dump(best_model, os.path.join(model_dir, 'best_model.pkl'))
    print(f"\n✅ Best Model: {best_model_name} with ROC AUC = {best_auc:.4f} (Saved to models/best_model.pkl)")
