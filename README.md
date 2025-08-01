 Customer Churn Analysis - Machine Learning Project

This project predicts customer churn using various machine learning models. It is structured and modularized for maintainability and scalability, with components like preprocessing, feature engineering, model training, and evaluation.

📁 Folder Structure
churn_analysis/
│
├── data/ # Raw input data (churn_analysis.csv)
│ └── churn_analysis.csv
│
├── src/ # Source code
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── model.py
│ └── evaluation.py
│
├── notebooks/ # Jupyter notebooks (exploratory)
│ ├── 01_EDA.ipynb
│ ├── 02_Feature_Engineering.ipynb
│ └── 03_Model_Building.ipynb
│
├── models/ # Trained model pickle files
│
├── output/ # Output reports, plots, etc.
│
├── churnanalysis.db # SQLite DB (optional if used)
├── churnanalysis.sqlite # Alternate DB format (optional)
├── main.py # Main entry point for running full pipeline
└── README.md # Project documentation


 Project Highlights

- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Scikit-learn, XGBoost
- **Models used**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
- **Evaluation**:
  - Accuracy
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

