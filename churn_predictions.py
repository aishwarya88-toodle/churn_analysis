import pandas as pd

# Predict on test data
y_pred = pipeline.predict(X_test)

# Save predictions to CSV
output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output_df.to_csv('output/churn_predictions.csv', index=False)
