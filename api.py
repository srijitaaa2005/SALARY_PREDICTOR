from flask import Flask, request, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = os.path.join('models', 'salary_model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from the form
        age = float(request.form['age'])
        job_title = request.form['job_title']
        education_level = request.form['education_level']
        gender = request.form['gender']
        years_exp = float(request.form['years_experience'])

        # Create a DataFrame with correct column names
        input_data = pd.DataFrame([{
            'Age': age,
            'Job Title': job_title,
            'Education Level': education_level,
            'Gender': gender,
            'Years of Experience': years_exp
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template(
            'index.html',
            prediction_text=f'Predicted Salary: ₹{round(prediction, 2):,.2f}'
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
