from flask import Flask, render_template, request
import numpy as np
import joblib

#Flask App Setup
app = Flask(__name__)

# Load the saved model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("train_model.pkl")

#Home route for main page
@app.route('/')
def home():
    return render_template('index.html')

#Prediction route for form submission
@app.route('/submit', methods=['POST'])
def predict_personality():
    try:
        if request.method == 'POST':
            # Map Gender to numerical value
            Gender = request.form['Gender']
            Gender_no = 1 if Gender == "Female" else 2

            Age = float(request.form['Age'])
            openness = float(request.form['openness'])
            neuroticism = float(request.form['neuroticism'])
            conscientiousness = float(request.form['conscientiousness'])
            agreeableness = float(request.form['agreeableness'])
            extraversion = float(request.form['extraversion'])
            
            # Server-side validation for non-Age columns
            if not (1 <= openness <= 8 and 1 <= neuroticism <= 8 and 
                    1 <= conscientiousness <= 8 and 1 <= agreeableness <= 8 and 1 <= extraversion <= 8):
                return "Input values for openness, neuroticism, conscientiousness, agreeableness, and extraversion must be between 1 and 8", 400

            # Create the feature array
            input_data = np.array([[Gender_no, Age, openness, neuroticism, conscientiousness, agreeableness, extraversion]])

            # Use the loaded scaler to transform the input
            scaled_input_data = scaler.transform(input_data)
            print(scaled_input_data)

            # Predict the personality
            personality = str(model.predict(scaled_input_data)[0])

            return render_template('submit.html', answer=personality)
    except Exception as e:
        print(f"Error: {e}")
        return None

#Run the flask app
if __name__ == "__main__":
    app.run(debug=True)
