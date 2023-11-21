import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

try:
    # Get the current directory
    current_directory = os.getcwd()

    # Navigate to the parent folder
    parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

    # Build the path to the CSV file in the "loan_risk/model" folder
    csv_path = os.path.join(parent_directory, 'model', 'credit.csv')

    # Load the dataset from the CSV file
    df = pd.read_csv(csv_path)

    # Split the data into features (X) and labels (y)
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the categorical and numeric columns
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    numeric_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

    # Create a column transformer for data preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    # Create the RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)

    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('Confusion Matrix:\n', conf_matrix)
    print('Classification Report:\n', classification_rep)

except FileNotFoundError:
    print("Error: CSV file not found.")
except Exception as e:
    print(f"Error: {str(e)}")





