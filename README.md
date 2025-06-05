# Personality-Prediction-Using-Machine-Learning

![image](https://github.com/user-attachments/assets/3555911e-291f-44e7-868c-701d4c116489)

**Dataset Overview**

There are two datasets provided for this analysis. The Train Dataset contains 709 records and is used to train a machine learning model to predict an individual's personality type based on various psychological traits and demographic features. The Test Dataset contains 315 records and is used to evaluate the trained model's performance on previously unseen data, ensuring its ability to generalize well to new inputs. Both datasets are structured and include a mix of numerical and categorical variables, making them suitable for classification tasks in personality prediction.

| Column Name                 | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| `Gender`                    | Gender of the individual (Male/Female).                               |
| `Age`                       | Age of the individual in years.                                       |
| `openness`                  | Score indicating openness to experience (1 to 8 scale).               |
| `neuroticism`               | Score indicating emotional instability or neuroticism (1 to 8 scale). |
| `conscientiousness`         | Score indicating self-discipline and organization (1 to 8 scale).     |
| `agreeableness`             | Score indicating compassion and cooperativeness (1 to 8 scale).       |
| `extraversion`              | Score indicating sociability and outgoing nature (1 to 8 scale).      |
| `Personality (Class label)` | The personality type label (e.g., serious, confident, nervous, etc.). |


**🧠 Personality Prediction Using Machine Learning**

This project aims to predict an individual's personality type based on psychological traits and demographic data using a Machine Learning model. It includes end-to-end implementation: from data preprocessing and model training to a user-friendly web interface built with Flask for real-time predictions.

**💡 Features**

Preprocessed Dataset: Gender encoded numerically, and features standardized using **StandardScaler**.

Model Training: **Logistic Regression** optimized via **GridSearchCV** for best parameters.

Model Evaluation: Evaluated on a separate test dataset.

Web App: Built using **Flask** with interactive user input form and result display.

Model Persistence: **joblib** used to save and load the trained model and scaler.

**🚀 How It Works**

User inputs gender, age, and trait scores (1–8 scale).

Input is transformed using the pre-fitted scaler.

Trained model predicts the personality type.

Result is shown to the user on a new page.

**📈 Model Performance**

Test Accuracy: 79%
