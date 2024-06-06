import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load dataset
file_path = 'Cardio_Data_fmcs.csv'  # Ensure this file exists in the same directory or provide the correct path
dataset = pd.read_csv(file_path)

# Print column names for verification
print("Columns in the dataset:", dataset.columns)

# Convert age from years to days
dataset['age'] = dataset['age'] * 365

# Define the target variable
y = dataset['target']

# Check for and handle missing values in the target column
if y.isna().sum() > 0:
    print(f"Missing values found in target column: {y.isna().sum()}")
    # Dropping rows with missing target values
    dataset = dataset.dropna(subset=['target'])
    y = dataset['target']

# Defining the features and the target variable
x = dataset[['age', 'height', 'weight', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']]

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Calculate accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Training data accuracy:", training_data_accuracy)


# Save the trained model as a pickle file
model_path = 'heart_disease_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")
