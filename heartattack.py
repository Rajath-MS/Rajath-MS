# Import necessary libraries
import sys
from pyscript import display

display(sys.version)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "heartattack.csv"
df = pd.read_csv(file_path)

# Display the first few rows and dataset info to understand its structure
print("First few rows of the dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nChecking for missing values:")
missing_values = df.isnull().sum()
print(missing_values)

# Fill missing values with column means if any are present
if missing_values.sum() > 0:
    df.fillna(df.mean(), inplace=True)
    print("Missing values have been filled with column mean values.")

# Define features (X) and target (y)
if 'target' in df.columns:
    X = df.drop('target', axis=1)
    y = df['target']
else:
    print("The target column 'target' is not found in the dataset. Please check column names.")
    exit()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("\nData successfully split into training and testing sets.")

# List of models to evaluate
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=10000)
}

# Dictionary to store the accuracy of each model
model_accuracies = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# Plotting the accuracies
model_names = list(model_accuracies.keys())
accuracies = list(model_accuracies.values())

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies)
plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()
