#importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

#load dataset
data = pd.read_csv('train dataset.csv')

#data preprocessing
data['Gender'] = data['Gender'].map({'Male': 2, 'Female': 1})
input_cols = ['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality (Class label)']
print("data",data['Gender'])

scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])
print("scaler",data[input_cols])

#model training
X = data[input_cols]
Y = data[output_cols].squeeze()

# Hyperparameter tuning using GridSearchCV for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'saga'],
    'max_iter': [100, 500, 1000]}

model = GridSearchCV(LogisticRegression(multi_class='multinomial'), param_grid, cv=5)
model.fit(X, Y)

# Get the best model parameters
best_params = model.best_params_
print("Best Parameters:", best_params)

# Train the model with the best parameters
best_model = LogisticRegression(**best_params, multi_class='multinomial')
best_model.fit(X, Y)


# Save the trained model and scaler
joblib.dump(best_model, "train_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Load and preprocess test data
test_data = pd.read_csv('test dataset.csv')
test_data['Gender'] = test_data['Gender'].map({'Male': 2, 'Female': 1})
test_data[input_cols] = scaler.transform(test_data[input_cols])

# Model testing
X_test = test_data[input_cols]
Y_test = test_data['Personality (class label)']
y_pred = best_model.predict(X_test)

# Model evaluation

# Cross-validation for performance estimation
cv_scores = cross_val_score(best_model, X, Y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean Cross-validation score:", cv_scores.mean())

score = accuracy_score(Y_test, y_pred) * 100
print("Accuracy_score:", score)
print("Best Parameters:", best_params)
